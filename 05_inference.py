"""Score engineered sensor features with a saved VedaSync model bundle."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "risk_model_bundle.joblib"
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_demo_features.csv"
BASE_SENSOR_FIELDS = {
    f"{metric}_{site}_{side}"
    for metric in ("temperature_celsius", "pressure_grams", "moisture_rh_percent", "spo2_percent")
    for site in ("Heel", "Metatarsal", "Toe", "Midfoot", "Big_Toe")
    for side in ("Left", "Right")
}


class VedaSyncPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.encoder = self.bundle["label_encoder"]
        self.raw_features = self.bundle["raw_feature_names"]
        self.imputer = self.bundle["imputer"]
        self.scaler = self.bundle["scaler"]
        self.needs_scaling = self.bundle["needs_scaling"]
        self.medians = self.bundle["training_medians"]

    def _model_input(self, values: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(values)
        return self.scaler.transform(imputed) if self.needs_scaling else imputed

    def predict(self, feature_dict: dict) -> dict:
        row = pd.DataFrame([{name: feature_dict.get(name, np.nan) for name in self.raw_features}])
        row = row.apply(pd.to_numeric, errors="coerce")
        missing_sensor_fields = sorted(
            name for name in BASE_SENSOR_FIELDS
            if name not in feature_dict or pd.isna(feature_dict.get(name))
        )
        quality_value = feature_dict.get("data_quality_flag", False)
        quality_flag = False if pd.isna(quality_value) else bool(quality_value)
        missing_count_value = feature_dict.get("sensor_missing_count", 0)
        missing_count = 0 if pd.isna(missing_count_value) else int(missing_count_value)
        quality_flag = quality_flag or bool(missing_count) or bool(missing_sensor_fields)

        model_input = self._model_input(row)
        probabilities = self.model.predict_proba(model_input)[0]
        class_index = int(np.argmax(probabilities))
        category = str(self.encoder.inverse_transform([class_index])[0])
        probability_by_class = {
            str(label): float(probability)
            for label, probability in zip(self.encoder.classes_, probabilities)
        }
        risk_score = 100 * (
            probability_by_class.get("High", 0.0)
            + 0.5 * probability_by_class.get("Moderate", 0.0)
        )
        entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1))))

        if hasattr(self.model, "feature_importances_"):
            importance = np.asarray(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            importance = np.ones(len(self.raw_features))
        importance_by_feature = dict.fromkeys(self.raw_features, 0.0)
        for name, value in zip(self.bundle["model_feature_names"], importance):
            feature = name.removeprefix("missingindicator_")
            if feature in importance_by_feature:
                importance_by_feature[feature] += float(value)
        candidates = sorted(importance_by_feature, key=importance_by_feature.get)
        candidates = candidates[-min(12, len(candidates)):]
        base_probability = probabilities[class_index]
        factors = []
        for feature in candidates:
            counterfactual = row.copy()
            counterfactual.loc[0, feature] = self.medians.get(feature, 0.0)
            changed_probability = self.model.predict_proba(self._model_input(counterfactual))[0, class_index]
            impact = float(base_probability - changed_probability)
            factors.append({
                "feature": feature,
                "value": None if pd.isna(row.loc[0, feature]) else float(row.loc[0, feature]),
                "impact": impact,
                "direction": "increases" if impact > 1e-6 else "decreases" if impact < -1e-6 else "neutral",
            })
        factors.sort(key=lambda factor: abs(factor["impact"]), reverse=True)

        return {
            "risk_score": round(float(risk_score), 2),
            "category": category,
            "top_factors": factors[:5],
            "uncertainty": {
                "confidence": float(probabilities[class_index]),
                "entropy": entropy,
                "calibration": "uncalibrated model probability",
            },
            "data_quality_flag": quality_flag,
            "data_quality": {
                "missing_sensor_fields": missing_sensor_fields,
                "sensor_missing_count": max(missing_count, len(missing_sensor_fields)),
            },
            "probabilities": probability_by_class,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FEATURE_PATH, help="CSV of engineered feature rows")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--row", type=int, default=-1, help="Zero-based input row (default: latest row)")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError(f"No feature rows available in {args.input}")
    row = frame.iloc[args.row]
    result = VedaSyncPredictor().predict(row.to_dict())
    result["patient_id"] = row.get("patient_id")
    result["device_id"] = row.get("device_id")
    result["timestamp"] = str(row.get("timestamp"))
    result["data_source"] = row.get("data_source", "UNKNOWN")
    print(pd.Series(result).to_json(indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pd.Series(result).to_json(indent=2), encoding="utf-8")
        print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
