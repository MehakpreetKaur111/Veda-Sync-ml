"""
VedaSync - Step 4: Train Multi-Class Risk Model
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "risk_level"

DROP_COLS = [
    "patient_id", "patient_name", "timestamp", "day_of_week",
    "risk_level", "risk_score",
    "ulcer_risk", "neuropathy_risk",
    "alert_triggered", "alert_type",
    "reading_quality", "medical_condition",
]


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET].astype(str)

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    groups = df["patient_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y_enc[train_idx]
    y_test  = y_enc[test_idx]

    scaler = StandardScaler().fit(X_train)

    return X_train, X_test, y_train, y_test, le, scaler, X.columns.tolist()


def train_all_models(X_train, y_train, scaler):
    try:
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print(f"📊 After SMOTE: {X_train_bal.shape[0]:,} samples")
    except Exception as e:
        print(f"⚠️  SMOTE failed ({e}). Using original training data.")
        X_train_bal, y_train_bal = X_train, y_train

    X_train_s_bal = scaler.transform(X_train_bal)

    models = {}

    print("\n🔹 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_s_bal, y_train_bal)
    models["LogisticRegression"] = {"model": lr, "needs_scaling": True}

    print("🔹 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    models["RandomForest"] = {"model": rf, "needs_scaling": False}

    print("🔹 Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train_bal, y_train_bal)
    models["XGBoost"] = {"model": xgb, "needs_scaling": False}

    return models


def evaluate_models(models, X_test, X_test_s, y_test, le):
    results = {}
    for name, entry in models.items():
        model = entry["model"]
        X_eval = X_test_s if entry["needs_scaling"] else X_test

        y_pred = model.predict(X_eval)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        try:
            roc = roc_auc_score(
                y_test, model.predict_proba(X_eval),
                multi_class="ovr", average="weighted"
            )
        except Exception:
            roc = None

        results[name] = {
            "accuracy":    round(float(acc), 4),
            "weighted_f1": round(float(f1), 4),
            "roc_auc":     round(float(roc), 4) if roc is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[str(c) for c in le.classes_],
                output_dict=True, zero_division=0
            ),
        }

        print(f"✅ {name:20s} | acc={acc:.4f} | F1={f1:.4f} | "
              f"ROC-AUC={roc if roc is None else round(roc, 4)}")

    return results


def cross_validate(model, X, y, name, cv=5):
    print(f"\n🔁 Cross-validating {name} (cv={cv})...")
    scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring="f1_weighted", n_jobs=-1
    )
    print(f"   CV F1 scores: {scores.round(4)} | Mean: {scores.mean():.4f}")
    return scores.tolist()


if __name__ == "__main__":
    df = pd.read_csv(FEATURE_PATH)
    print(f"📥 Loaded: {df.shape}")
    print(f"📊 Risk level distribution:\n{df['risk_level'].value_counts()}")

    X_train, X_test, y_train, y_test, le, scaler, feature_names = prepare_data(df)

    print(f"\n📊 Train size: {len(X_train):,}")
    print(f"📊 Test  size: {len(X_test):,}")
    print(f"📊 Classes  : {list(le.classes_)}")

    X_test_s = scaler.transform(X_test)

    models  = train_all_models(X_train, y_train, scaler)
    results = evaluate_models(models, X_test, X_test_s, y_test, le)

    best_name = max(results, key=lambda k: results[k]["weighted_f1"])
    print(f"\n🏆 Best model: {best_name} "
          f"(F1 = {results[best_name]['weighted_f1']})")

    best_entry = models[best_name]
    if best_entry["needs_scaling"]:
        cv_scores = cross_validate(
            best_entry["model"], scaler.transform(X_train), y_train, best_name
        )
    else:
        cv_scores = cross_validate(
            best_entry["model"], X_train, y_train, best_name
        )

    joblib.dump(best_entry["model"],    MODEL_DIR / "risk_model.pkl")
    joblib.dump(best_entry["needs_scaling"], MODEL_DIR / "needs_scaling.pkl")
    joblib.dump(scaler,                 MODEL_DIR / "scaler.pkl")
    joblib.dump(le,                     MODEL_DIR / "label_encoder.pkl")
    joblib.dump(feature_names,          MODEL_DIR / "feature_names.pkl")

    with open(OUT_DIR / "model_metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "cv_f1": cv_scores,
            "results": results,
        }, f, indent=2)

    print(f"\n💾 Saved models → {MODEL_DIR}/")
    print(f"💾 Saved metrics → {OUT_DIR}/model_metrics.json")