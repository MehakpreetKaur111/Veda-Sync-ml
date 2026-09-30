"""
VedaSync - Step 3: Feature Engineering
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "veda_sync_clean.csv"
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_features.csv"


def safe_mode(series, default="Unknown"):
    m = series.dropna().mode()
    return m.iloc[0] if not m.empty else default


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- 1. Pivot sensors wide ----------
    pivot = df.pivot_table(
        index=["patient_id", "patient_name", "medical_condition",
               "patient_age", "timestamp", "day_of_week", "hour_of_day"],
        columns="sensor_location",
        values=["temperature_celsius", "pressure_grams",
                "moisture_rh_percent", "spo2_percent"],
        aggfunc="mean"
    )
    pivot.columns = [f"{val}_{loc}" for val, loc in pivot.columns]
    pivot = pivot.reset_index()

    # ---------- 2. Aggregate non-sensor features ----------
    agg_cols = {
        "heart_rate_bpm":         "mean",
        "humidity_level":         "mean",
        "device_battery_percent": "mean",
        "risk_level":             lambda x: safe_mode(x, "Low"),
        "ulcer_risk":             lambda x: safe_mode(x, "Low"),
        "neuropathy_risk":        lambda x: safe_mode(x, "Low"),
        "alert_triggered":        lambda x: "Yes" if "Yes" in x.values else "No",
        "reading_quality":        lambda x: safe_mode(x, "Good"),
    }
    agg = df.groupby(["patient_id", "timestamp"]).agg(agg_cols).reset_index()

    merged = pivot.merge(agg, on=["patient_id", "timestamp"], how="left")

    # ---------- 3. Derived clinical features ----------
    if "temperature_celsius_Heel" in merged and "temperature_celsius_Big_Toe" in merged:
        merged["temp_asymmetry_heel_toe"] = (
            merged["temperature_celsius_Heel"] - merged["temperature_celsius_Big_Toe"]
        ).abs()

    if "temperature_celsius_Metatarsal" in merged and "temperature_celsius_Heel" in merged:
        merged["temp_asymmetry_met_heel"] = (
            merged["temperature_celsius_Metatarsal"] - merged["temperature_celsius_Heel"]
        ).abs()

    temp_cols = [c for c in merged.columns if c.startswith("temperature_celsius_")]
    if temp_cols:
        merged["temp_mean"]  = merged[temp_cols].mean(axis=1)
        merged["temp_max"]   = merged[temp_cols].max(axis=1)
        merged["temp_min"]   = merged[temp_cols].min(axis=1)
        merged["temp_range"] = merged["temp_max"] - merged["temp_min"]

    pres_cols = [c for c in merged.columns if c.startswith("pressure_grams_")]
    if pres_cols:
        merged["pressure_mean"] = merged[pres_cols].mean(axis=1)
        merged["pressure_max"]  = merged[pres_cols].max(axis=1)

    if "pressure_grams_Heel" in merged and "pressure_grams_Toe" in merged:
        merged["pressure_heel_toe_ratio"] = (
            merged["pressure_grams_Heel"] / (merged["pressure_grams_Toe"] + 1e-3)
        )

    moist_cols = [c for c in merged.columns if c.startswith("moisture_rh_percent_")]
    if moist_cols:
        merged["moisture_mean"] = merged[moist_cols].mean(axis=1)

    spo2_cols = [c for c in merged.columns if c.startswith("spo2_percent_")]
    if spo2_cols:
        merged["spo2_mean"] = merged[spo2_cols].mean(axis=1)
        merged["spo2_min"]  = merged[spo2_cols].min(axis=1)

    merged["low_battery_flag"]  = (merged["device_battery_percent"] < 20).astype(int)
    merged["poor_quality_flag"] = (merged["reading_quality"] == "Poor").astype(int)

    # ---------- 4. Personal baseline deviation ----------
    merged = merged.sort_values(["patient_id", "timestamp"])
    baseline_cols = ["temp_mean", "pressure_mean", "moisture_mean", "spo2_mean"]
    for col_name in baseline_cols:
        if col_name not in merged.columns:
            continue
        baseline = (
            merged.groupby("patient_id")[col_name]
                  .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )
        merged[f"{col_name}_baseline"]  = baseline
        merged[f"{col_name}_deviation"] = merged[col_name] - baseline

    # ---------- 5. Time-of-day context ----------
    merged["is_night"] = merged["hour_of_day"].isin(
        [22, 23, 0, 1, 2, 3, 4, 5]
    ).astype(int)

    print(f"✅ Engineered features: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    return merged


if __name__ == "__main__":
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    feats = engineer_features(df)
    feats.to_csv(FEATURE_PATH, index=False)
    print(f"💾 Saved features → {FEATURE_PATH}")
    print(f"\nColumns ({len(feats.columns)}):\n{list(feats.columns)}")