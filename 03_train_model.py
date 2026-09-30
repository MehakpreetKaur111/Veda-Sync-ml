"""Compare risk classifiers using a device-held-out test split."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_demo_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
TARGET = "risk_level"
EXCLUDED_COLUMNS = {
    "patient_id", "device_id", "timestamp", "day_of_week", "hour_of_day",
    "risk_level", "risk_score", "ulcer_risk", "neuropathy_risk",
    "alert_triggered", "alert_type", "reading_quality", "medical_condition",
    "data_source", "simulated_ulcer_event_time", "data_quality_flag",
}


def select_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    candidates = frame.drop(columns=[column for column in EXCLUDED_COLUMNS if column in frame])
    numeric = candidates.select_dtypes(include=[np.number, "bool"]).copy()
    if numeric.empty:
        raise ValueError("No numeric model features found in the engineered data.")
    return numeric, numeric.columns.tolist()


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y)
    smallest_class = int(counts[counts > 0].min())
    if len(counts[counts > 0]) < 2 or smallest_class < 2:
        print("SMOTE skipped: at least two samples per training class are required.")
        return X, y
    sampler = SMOTE(random_state=42, k_neighbors=min(5, smallest_class - 1))
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    print(f"SMOTE training rows: {len(X):,} -> {len(X_balanced):,}")
    return X_balanced, y_balanced


def main() -> None:
    frame = pd.read_csv(FEATURE_PATH, parse_dates=["timestamp"])
    if "device_id" not in frame or TARGET not in frame:
        raise ValueError("Feature data must include device_id and risk_level.")
    X, raw_feature_names = select_features(frame)
    labels = frame[TARGET].astype(str)
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=frame["device_id"]))
    train_devices = set(frame.iloc[train_idx]["device_id"])
    test_devices = set(frame.iloc[test_idx]["device_id"])
    if train_devices & test_devices:
        raise RuntimeError("Device leakage detected between training and test data.")

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    X_train = imputer.fit_transform(X.iloc[train_idx])
    X_test = imputer.transform(X.iloc[test_idx])
    model_feature_names = imputer.get_feature_names_out(raw_feature_names).tolist()
    medians = pd.Series(X.iloc[train_idx].median(), index=raw_feature_names).fillna(0)
    X_train_balanced, y_train_balanced = apply_smote(X_train, y[train_idx])
    scaler = StandardScaler().fit(X_train_balanced)
    X_train_scaled = scaler.transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    estimators = {
        "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight=None, random_state=42), True),
        "RandomForest": (RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight=None,
            random_state=42, n_jobs=-1,
        ), False),
        "XGBoost": (XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            objective="multi:softprob", num_class=len(encoder.classes_),
            random_state=42, n_jobs=-1,
        ), False),
    }
    models = {}
    results = {}
    predictions = {}
    for name, (model, needs_scaling) in estimators.items():
        training_data = X_train_scaled if needs_scaling else X_train_balanced
        testing_data = X_test_scaled if needs_scaling else X_test
        model.fit(training_data, y_train_balanced)
        predicted = model.predict(testing_data).astype(int)
        probabilities = model.predict_proba(testing_data)
        report = classification_report(
            y[test_idx], predicted, labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_.tolist(), output_dict=True, zero_division=0,
        )
        results[name] = {
            "precision_macro": float(precision_score(y[test_idx], predicted, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y[test_idx], predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[test_idx], predicted, average="weighted", zero_division=0)),
            "per_class": report,
        }
        models[name] = {"model": model, "needs_scaling": needs_scaling}
        predictions[name] = (predicted, probabilities)
        print(f"{name}: macro recall={results[name]['recall_macro']:.3f}, "
              f"macro precision={results[name]['precision_macro']:.3f}, "
              f"weighted F1={results[name]['weighted_f1']:.3f}")

    best_name = max(results, key=lambda name: (
        results[name]["recall_macro"], results[name]["weighted_f1"], results[name]["precision_macro"]
    ))
    best_model = models[best_name]["model"]
    best_needs_scaling = models[best_name]["needs_scaling"]
    best_prediction, best_probabilities = predictions[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "needs_scaling": best_needs_scaling,
        "scaler": scaler,
        "imputer": imputer,
        "label_encoder": encoder,
        "raw_feature_names": raw_feature_names,
        "model_feature_names": model_feature_names,
        "training_medians": medians.to_dict(),
    }
    joblib.dump(bundle, MODEL_DIR / "risk_model_bundle.joblib")
    # Compatibility artifacts for the existing 07_predict.py entry point.
    joblib.dump(best_model, MODEL_DIR / "risk_model.pkl")
    joblib.dump(best_needs_scaling, MODEL_DIR / "needs_scaling.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(raw_feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")

    test_rows = frame.iloc[test_idx][["device_id", "timestamp", TARGET]].copy()
    test_rows["predicted_risk_level"] = encoder.inverse_transform(best_prediction)
    for class_index, class_name in enumerate(encoder.classes_):
        test_rows[f"probability_{class_name}"] = best_probabilities[:, class_index]
    if "simulated_ulcer_event_time" in frame:
        test_rows["simulated_ulcer_event_time"] = frame.iloc[test_idx]["simulated_ulcer_event_time"].values
    test_rows.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "training_data_source": "Synthetic DEMO DATA generated by 00_generate_demo_data.py",
        "split": "GroupShuffleSplit by device_id; 75% train / 25% held-out devices",
        "train_device_count": len(train_devices),
        "test_device_count": len(test_devices),
        "best_model": best_name,
        "selection_criterion": "highest macro recall, then weighted F1, then macro precision",
        "class_labels": encoder.classes_.tolist(),
        "results": results,
    }
    with (OUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"Selected {best_name}; saved model bundle and grouped test predictions.")


if __name__ == "__main__":
    main()
