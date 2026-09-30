# Data Schema

The project has no clinical dataset. `veda_sync_demo.csv` is generated synthetic data and is marked `data_source=DEMO DATA` on every row. It is for pipeline demonstration and software testing only; it is not patient data and must not be used for clinical decisions.

Older `veda_sync.csv`, `veda_sync_clean.csv`, and `veda_sync_features.csv` files are legacy artifacts and are not read by the current `run_all.py` pipeline. Their provenance is not established; do not treat them as clinical data.

## Raw Sensor CSV

One row is one sensor reading. A measurement time has five anatomical sites on each foot (`sensor_location` plus `side`) and shares `patient_id`, `device_id`, and `timestamp`.

| Column | Meaning |
|---|---|
| `patient_id`, `device_id` | Synthetic subject/device identifiers; the model split groups by `device_id`. |
| `timestamp` | ISO-8601 sensor collection time, UTC-naive in the demo. |
| `sensor_location` | `Heel`, `Metatarsal`, `Toe`, `Midfoot`, or `Big_Toe`. |
| `side` | `Left` or `Right`. |
| `temperature_celsius` | Skin temperature, degrees Celsius. |
| `pressure_grams` | Sensor pressure/load reading, grams. |
| `moisture_rh_percent` | Local relative humidity, percent. |
| `spo2_percent` | Simulated peripheral oxygen saturation, percent. |
| `heart_rate_bpm` | Simulated heart rate, beats per minute. |
| `humidity_level`, `device_battery_percent` | Simulated environmental humidity and remaining battery, percent. |
| `risk_level` | Simulated target class: `Low`, `Moderate`, or `High`, derived from progression and sensor thresholds. |
| `simulated_ulcer_event_time` | Synthetic future event time for lead-time evaluation; empty for non-progressing sequences. Not a model input. |
| `data_source` | Always `DEMO DATA` for generated records. |
| `reading_quality` | `Good` or `Poor`; may be poor when a sensor is missing. |

Sensor values may be missing. Cleaning preserves null values and adds `sensor_missing_count` and `data_quality_flag`; it does not median-fill readings. Configured physiological bounds are in `sensor_config.json`.

## Engineered Rows

`veda_sync_demo_features.csv` has one row per device and timestamp. It includes paired-side temperature asymmetry, site aggregates, rolling 6-hour/24-hour/7-day summaries, rolling pressure-time exposure, past-only personal baselines and deviations, quality flags, the `risk_level` target, and event metadata. IDs, event metadata, and target/derived label columns are excluded from training.
