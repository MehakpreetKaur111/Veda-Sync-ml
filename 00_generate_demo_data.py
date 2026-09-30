"""Generate explicitly labeled synthetic VedaSync sensor data for demos."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "veda_sync_demo.csv"
SITES = ("Heel", "Metatarsal", "Toe", "Midfoot", "Big_Toe")


def generate_demo_data(
    patients: int = 60,
    days: int = 28,
    seed: int = 42,
    start: str = "2025-01-01",
    warning_threshold: float = 1.0,
    high_threshold: float = 2.2,
) -> pd.DataFrame:
    """Create longitudinal bilateral readings with simulated progression labels."""
    if patients < 4 or days < 14:
        raise ValueError("Use at least 4 patients and 14 days to support grouped evaluation.")

    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp(start)
    records = []
    times = pd.date_range(start_time, periods=days * 4, freq="6h")

    for patient_index in range(patients):
        patient_id = f"DEMO-P{patient_index + 1:04d}"
        device_id = f"DEMO-D{patient_index + 1:04d}"
        baseline_temp = rng.normal(32.8, 0.35)
        baseline_pressure = rng.normal(390, 55)
        baseline_moisture = rng.normal(48, 7)
        progresses = rng.random() < 0.38
        event_time = (
            start_time + pd.Timedelta(days=int(rng.integers(16, days - 2)))
            if progresses else pd.NaT
        )
        susceptible_site = SITES[int(rng.integers(0, len(SITES)))]

        for timestamp in times:
            day_index = (timestamp - start_time).total_seconds() / 86400
            days_to_event = (event_time - timestamp).total_seconds() / 86400 if progresses else np.inf
            progression = 1 / (1 + np.exp(np.clip(days_to_event - 7, -30, 30))) if progresses else 0.0
            asymmetry = max(0.0, 0.25 + 2.6 * progression + rng.normal(0, 0.22))
            if progresses and days_to_event <= 7:
                risk_level = "High" if asymmetry >= high_threshold else "Moderate"
            elif asymmetry >= high_threshold:
                risk_level = "High"
            elif asymmetry >= warning_threshold:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

            # Include intermittent elevated loading so the demo has non-temperature signals.
            load_burst = rng.random() < (0.15 + 0.35 * progression)
            for site_index, site in enumerate(SITES):
                site_pressure = max(40, baseline_pressure + rng.normal(0, 115))
                if site == susceptible_site and load_burst:
                    site_pressure += rng.uniform(180, 380)
                site_temp = baseline_temp + 0.12 * site_index + rng.normal(0, 0.25)
                site_moisture = np.clip(baseline_moisture + rng.normal(0, 8), 15, 85)

                for side in ("Left", "Right"):
                    side_shift = asymmetry / 2 if side == "Left" else -asymmetry / 2
                    sensor_values = {
                        "temperature_celsius": site_temp + side_shift,
                        "pressure_grams": site_pressure + rng.normal(0, 35),
                        "moisture_rh_percent": site_moisture + rng.normal(0, 3),
                        "spo2_percent": np.clip(97.5 - 2.0 * progression + rng.normal(0, 1), 90, 100),
                        "heart_rate_bpm": np.clip(76 + 8 * progression + rng.normal(0, 7), 48, 125),
                        "humidity_level": np.clip(52 + rng.normal(0, 9), 20, 85),
                        "device_battery_percent": np.clip(95 - day_index * 0.65 + rng.normal(0, 1.5), 5, 100),
                    }
                    for sensor_name in sensor_values:
                        if rng.random() < 0.012:
                            sensor_values[sensor_name] = np.nan

                    reading_quality = "Poor" if any(pd.isna(v) for v in sensor_values.values()) else "Good"
                    records.append({
                        "patient_id": patient_id,
                        "device_id": device_id,
                        "timestamp": timestamp,
                        "sensor_location": site,
                        "side": side,
                        **sensor_values,
                        "risk_level": risk_level,
                        "simulated_ulcer_event_time": event_time,
                        "data_source": "DEMO DATA",
                        "reading_quality": reading_quality,
                    })

    data = pd.DataFrame.from_records(records)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2025-01-01", help="First simulated date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = BASE_DIR / "sensor_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = config["clinical_thresholds"]
    data = generate_demo_data(
        args.patients, args.days, args.seed, args.start,
        thresholds["temperature_asymmetry_warning_c"],
        thresholds["temperature_asymmetry_high_c"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Generated DEMO DATA: {len(data):,} readings for {args.patients} devices -> {args.output}")
    print(data["risk_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
