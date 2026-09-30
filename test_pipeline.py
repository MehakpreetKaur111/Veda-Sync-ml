import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_test_bundle(path: Path) -> None:
    features = ["feature_a", "feature_b"]
    X = pd.DataFrame({"feature_a": [0, 1, 2, 3, 4, 5], "feature_b": [9, 8, 7, 6, 5, 4]})
    labels = np.array(["Low", "Moderate", "High", "Low", "Moderate", "High"])
    encoder = LabelEncoder().fit(labels)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(X)
    transformed = imputer.transform(X)
    scaler = StandardScaler().fit(transformed)
    model = RandomForestClassifier(n_estimators=8, random_state=42).fit(
        transformed, encoder.transform(labels)
    )
    joblib.dump({
        "model": model,
        "model_name": "RandomForest",
        "needs_scaling": False,
        "scaler": scaler,
        "imputer": imputer,
        "label_encoder": encoder,
        "raw_feature_names": features,
        "model_feature_names": imputer.get_feature_names_out(features).tolist(),
        "training_medians": X.median().to_dict(),
    }, path)


load_data_module = load_module("data_loading", "01_data_loading.py")
clean_data = load_data_module.clean_data
resolve_data_path = load_data_module.resolve_data_path
validate_schema = load_data_module.validate_schema
feature_module = load_module("feature_engineering", "02_feature_engineering.py")
engineer_features = feature_module.engineer_features
predictor_module = load_module("predictor", "07_predict.py")
VedaSyncPredictor = predictor_module.VedaSyncPredictor
inference_module = load_module("inference", "05_inference.py")
RiskPredictor = inference_module.VedaSyncPredictor


def test_resolve_data_path_uses_explicit_csv():
    path = resolve_data_path(ROOT / "data" / "veda_sync.csv")
    assert path == (ROOT / "data" / "veda_sync.csv").resolve()


def test_loader_preserves_missing_readings_and_flags_quality():
    row = {
        "patient_id": "p1", "device_id": "d1", "timestamp": "2025-01-01",
        "sensor_location": "Heel", "temperature_celsius": None,
        "pressure_grams": 200, "moisture_rh_percent": 40,
        "spo2_percent": 98, "risk_level": "Low", "reading_quality": "Good",
    }
    cleaned = clean_data(pd.DataFrame([row]), {"clamp_ranges": {"temperature_celsius": [20, 45]}})
    assert pd.isna(cleaned.loc[0, "temperature_celsius"])
    assert cleaned.loc[0, "sensor_missing_count"] == 1
    assert cleaned.loc[0, "data_quality_flag"]


def test_loader_rejects_missing_schema_columns():
    with pytest.raises(ValueError, match="missing required columns.*device_id"):
        validate_schema(pd.DataFrame({"patient_id": ["p1"]}))


def test_feature_engineering_builds_windows_and_past_baseline():
    records = []
    for step, timestamp in enumerate(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 06:00"])):
        for site in ("Heel", "Big_Toe"):
            for side in ("Left", "Right"):
                records.append({
                    "patient_id": "p1", "device_id": "d1", "timestamp": timestamp,
                    "sensor_location": site, "side": side,
                    "temperature_celsius": 34 + (2.4 if step == 1 and side == "Left" else 0),
                    "pressure_grams": 650, "moisture_rh_percent": 45,
                    "spo2_percent": 98, "heart_rate_bpm": 75,
                    "humidity_level": 50, "device_battery_percent": 90,
                    "risk_level": "High" if step else "Low",
                    "sensor_missing_count": 0, "data_quality_flag": False,
                    "reading_quality": "Good", "data_source": "DEMO DATA",
                })
    features = engineer_features(pd.DataFrame(records))
    second = features.iloc[1]
    assert second["temp_asymmetry"] == pytest.approx(2.4)
    assert second["temp_mean_baseline"] == pytest.approx(34.0)
    assert second["pressure_time_exposure_6h"] == pytest.approx(900)
    assert "temp_mean_mean_24h" in features
    assert "pressure_max_max_7d" in features


def test_predictor_accepts_feature_dataframe_columns(tmp_path):
    bundle_path = tmp_path / "bundle.joblib"
    make_test_bundle(bundle_path)
    predictor = VedaSyncPredictor(bundle_path)
    sample = {
        'temperature_celsius_Heel': 36.5,
        'temperature_celsius_Metatarsal': 36.0,
        'temperature_celsius_Toe': 35.2,
        'temperature_celsius_Midfoot': 34.5,
        'temperature_celsius_Big_Toe': 34.8,
        'pressure_grams_Heel': 450,
        'pressure_grams_Metatarsal': 220,
        'pressure_grams_Toe': 300,
        'pressure_grams_Midfoot': 200,
        'pressure_grams_Big_Toe': 180,
        'moisture_rh_percent_Heel': 55,
        'moisture_rh_percent_Metatarsal': 50,
        'moisture_rh_percent_Toe': 48,
        'moisture_rh_percent_Midfoot': 52,
        'moisture_rh_percent_Big_Toe': 47,
        'spo2_percent_Heel': 98,
        'spo2_percent_Metatarsal': 97,
        'spo2_percent_Toe': 98,
        'spo2_percent_Midfoot': 97,
        'spo2_percent_Big_Toe': 97,
        'heart_rate_bpm': 78,
        'humidity_level': 55,
        'device_battery_percent': 80,
        'temp_asymmetry_heel_toe': 1.7,
        'temp_mean': 35.4,
        'temp_max': 36.5,
        'temp_range': 2.0,
        'pressure_mean': 270,
        'pressure_max': 450,
        'moisture_mean': 50.4,
        'spo2_mean': 97.4,
        'spo2_min': 97,
        'low_battery_flag': 0,
        'poor_quality_flag': 0,
        'is_night': 0,
        'temp_mean_baseline': 35.3,
        'temp_mean_deviation': 0.1,
        'pressure_mean_baseline': 260,
        'pressure_mean_deviation': 10,
        'moisture_mean_baseline': 50,
        'moisture_mean_deviation': 0.4,
        'spo2_mean_baseline': 97.5,
        'spo2_mean_deviation': -0.1,
    }
    result = predictor.predict(sample)
    assert set(result) == {'risk_level', 'confidence', 'probabilities'}
    assert result['risk_level'] in predictor.le.classes_
    assert 0.0 <= result['confidence'] <= 1.0


def test_inference_returns_readme_schema_and_quality_flag(tmp_path):
    bundle_path = tmp_path / "bundle.joblib"
    make_test_bundle(bundle_path)
    result = RiskPredictor(bundle_path).predict({"feature_a": 2.0, "feature_b": 7.0})
    assert {"risk_score", "category", "top_factors", "uncertainty", "data_quality_flag"} <= set(result)
    assert 0 <= result["risk_score"] <= 100
    assert result["category"] in {"Low", "Moderate", "High"}
    assert result["data_quality_flag"]
    sensor_values = {name: 32.0 for name in inference_module.BASE_SENSOR_FIELDS}
    sensor_values["temperature_celsius_Heel_Left"] = None
    sensor_values.update({"feature_a": 2.0, "feature_b": 7.0})
    null_result = RiskPredictor(bundle_path).predict(sensor_values)
    assert null_result["data_quality"]["missing_sensor_fields"] == ["temperature_celsius_Heel_Left"]
    assert null_result["data_quality"]["sensor_missing_count"] == 1
