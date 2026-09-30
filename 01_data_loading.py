"""Load and validate VedaSync sensor readings."""
import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "veda_sync_demo.csv"
OUTPUT_PATH = BASE_DIR / "data" / "veda_sync_demo_clean.csv"
CONFIG_PATH = BASE_DIR / "sensor_config.json"
REQUIRED_COLUMNS = {
    "patient_id", "device_id", "timestamp", "sensor_location",
    "temperature_celsius", "pressure_grams", "moisture_rh_percent",
    "spo2_percent", "risk_level",
}
SENSOR_COLUMNS = [
    "temperature_celsius", "pressure_grams", "moisture_rh_percent",
    "spo2_percent", "heart_rate_bpm", "humidity_level",
    "device_battery_percent",
]


def resolve_data_path(path: str | None = None) -> Path:
    """Resolve only the supplied path or the documented project default."""
    candidate = Path(path) if path else DATA_PATH
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {resolved}. Generate demo data with "
            "00_generate_demo_data.py or pass --input PATH."
        )
    return resolved


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
            + ". Required columns include patient_id, device_id, timestamp, "
            "sensor_location, temperature_celsius, pressure_grams, "
            "moisture_rh_percent, spo2_percent, and risk_level."
        )


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    resolved = resolve_data_path(str(path) if path else None)
    df = pd.read_csv(resolved)
    validate_schema(df)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {resolved}")
    return df


def clean_data(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    config = config or load_config()
    validate_schema(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates().copy()

    for col in ["sensor_location", "risk_level", "reading_quality", "medical_condition"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in SENSOR_COLUMNS + ["risk_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sensor_columns = [col for col in SENSOR_COLUMNS if col in df.columns]
    missing_counts = df[sensor_columns].isna().sum(axis=1)
    quality = df.get("reading_quality", pd.Series("Good", index=df.index))
    df["sensor_missing_count"] = missing_counts
    df["data_quality_flag"] = missing_counts.gt(0) | quality.str.lower().isin({"poor", "low"})

    for col, bounds in config["clamp_ranges"].items():
        if col in df.columns:
            df[col] = df[col].clip(float(bounds[0]), float(bounds[1]))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="Input sensor CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Cleaned CSV path")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Sensor limits JSON")
    args = parser.parse_args()

    cleaned = clean_data(load_data(args.input), load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Saved cleaned data to {args.output}")
    print(cleaned["risk_level"].value_counts())