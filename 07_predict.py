"""Compatibility interface for the original predictor response schema."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "risk_model_bundle.joblib"


class VedaSyncPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.needs_scaling = bundle["needs_scaling"]
        self.scaler = bundle["scaler"]
        self.imputer = bundle["imputer"]
        self.le = bundle["label_encoder"]
        self.features = bundle["raw_feature_names"]

    def predict(self, feature_dict: dict) -> dict:
        row = pd.DataFrame([{name: feature_dict.get(name, np.nan) for name in self.features}])
        row = row.apply(pd.to_numeric, errors="coerce")
        model_input = self.imputer.transform(row)
        if self.needs_scaling:
            model_input = self.scaler.transform(model_input)
        probabilities = self.model.predict_proba(model_input)[0]
        index = int(np.argmax(probabilities))
        return {
            "risk_level": self.le.inverse_transform([index])[0],
            "confidence": float(probabilities[index]),
            "probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.le.classes_, probabilities)
            },
        }


if __name__ == "__main__":
    features_path = BASE_DIR / "data" / "veda_sync_demo_features.csv"
    row = pd.read_csv(features_path).iloc[-1]
    print(VedaSyncPredictor().predict(row.to_dict()))