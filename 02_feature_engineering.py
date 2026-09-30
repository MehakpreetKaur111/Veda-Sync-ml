"""Aggregate sensor readings and derive longitudinal clinical demo features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "veda_sync_demo_clean.csv"
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_demo_features.csv"
SITES = ("Heel", "Metatarsal", "Toe", "Midfoot", "Big_Toe")
PAIRED_METRICS = ("temperature_celsius", "pressure_grams", "moisture_rh_percent", "spo2_percent")
ROLLING_METRICS = (
    "temp_mean", "temp_asymmetry", "pressure_mean", "pressure_max",
    "moisture_mean", "spo2_mean", "heart_rate_bpm",
)
BASELINE_METRICS = ("temp_mean", "pressure_mean", "moisture_mean", "spo2_mean")


def engineer_features(readings: pd.DataFrame, pressure_threshold: float = 500.0) -> pd.DataFrame:
    required = {"patient_id", "device_id", "timestamp", "sensor_location", "side", "risk_level"}
    missing = sorted(required - set(readings.columns))
    if missing:
        raise ValueError(f"Feature input is missing required columns: {', '.join(missing)}")

    data = readings.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    keys = ["patient_id", "device_id", "timestamp"]

    pivot = data.pivot_table(
        index=keys,
        columns=["sensor_location", "side"],
        values=list(PAIRED_METRICS),
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{site}_{side}" for metric, site, side in pivot.columns]
    expected = [
        f"{metric}_{site}_{side}"
        for metric in PAIRED_METRICS for site in SITES for side in ("Left", "Right")
    ]
    pivot = pivot.reindex(columns=expected).reset_index()

    metadata = {
        "risk_level": ("risk_level", "first"),
        "heart_rate_bpm": ("heart_rate_bpm", "mean"),
        "humidity_level": ("humidity_level", "mean"),
        "device_battery_percent": ("device_battery_percent", "mean"),
        "sensor_missing_count": ("sensor_missing_count", "sum"),
        "data_quality_flag": ("data_quality_flag", "max"),
        "reading_quality": ("reading_quality", lambda values: "Poor" if values.astype(str).str.lower().eq("poor").any() else "Good"),
        "data_source": ("data_source", "first"),
        "simulated_ulcer_event_time": ("simulated_ulcer_event_time", "first"),
    }
    aggregations = {
        output: spec for output, spec in metadata.items() if spec[0] in data.columns
    }
    summaries = data.groupby(keys, as_index=False).agg(**{
        output: (source, function) for output, (source, function) in aggregations.items()
    })
    features = pivot.merge(summaries, on=keys, how="left", validate="one_to_one")

    for metric in PAIRED_METRICS:
        for site in SITES:
            left = f"{metric}_{site}_Left"
            right = f"{metric}_{site}_Right"
            features[f"{metric}_{site}"] = features[[left, right]].mean(axis=1)

    temperature_differences = [
        (features[f"temperature_celsius_{site}_Left"] - features[f"temperature_celsius_{site}_Right"]).abs()
        for site in SITES
    ]
    features["temp_asymmetry"] = pd.concat(temperature_differences, axis=1).max(axis=1)
    features["temp_asymmetry_heel_toe"] = features["temp_asymmetry"]
    for metric, output in (
        ("temperature_celsius", "temp"),
        ("pressure_grams", "pressure"),
        ("moisture_rh_percent", "moisture"),
        ("spo2_percent", "spo2"),
    ):
        site_columns = [f"{metric}_{site}" for site in SITES]
        features[f"{output}_mean"] = features[site_columns].mean(axis=1)
        features[f"{output}_max"] = features[site_columns].max(axis=1)
        if output == "temp":
            features["temp_min"] = features[site_columns].min(axis=1)
            features["temp_range"] = features["temp_max"] - features["temp_min"]
        if output == "pressure":
            features["pressure_max"] = features[site_columns].max(axis=1)
        if output == "spo2":
            features["spo2_min"] = features[site_columns].min(axis=1)

    features["pressure_heel_toe_ratio"] = features["pressure_grams_Heel"] / (
        features["pressure_grams_Toe"] + 1e-3
    )
    features["pressure_excess_grams"] = (features["pressure_max"] - pressure_threshold).clip(lower=0)
    features["pressure_time_exposure_6h"] = features["pressure_excess_grams"] * 6
    features["poor_quality_flag"] = features["data_quality_flag"].fillna(True).astype(int)
    features["low_battery_flag"] = (features["device_battery_percent"] < 20).astype("Int64")
    features["is_night"] = features["timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

    features = features.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    for device_id, positions in features.groupby("device_id", sort=False).groups.items():
        device = features.loc[positions].set_index("timestamp")
        for hours, suffix in ((6, "6h"), (24, "24h"), (168, "7d")):
            rolling = device[list(ROLLING_METRICS)].rolling(f"{hours}h", min_periods=1).agg(["mean", "max"])
            rolling.columns = [f"{name}_{stat}_{suffix}" for name, stat in rolling.columns]
            features.loc[positions, rolling.columns] = rolling.to_numpy()
        for metric in BASELINE_METRICS:
            prior = device[metric].shift(1).expanding(min_periods=1).mean()
            features.loc[positions, f"{metric}_baseline"] = prior.to_numpy()
            features.loc[positions, f"{metric}_deviation"] = (
                device[metric].to_numpy() - prior.to_numpy()
            )

    return features


if __name__ == "__main__":
    frame = pd.read_csv(CLEAN_PATH)
    config = json.loads((BASE_DIR / "sensor_config.json").read_text(encoding="utf-8"))
    pressure_threshold = config["clinical_thresholds"]["pressure_alert_grams"]
    result = engineer_features(frame, pressure_threshold=pressure_threshold)
    result.to_csv(FEATURE_PATH, index=False)
    print(f"Saved {len(result):,} feature rows to {FEATURE_PATH}")
    print(f"Generated {len(result.columns)} columns")
