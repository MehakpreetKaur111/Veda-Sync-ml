"""
VedaSync - Step 6: Explainability
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "veda_sync_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = [
    "patient_id", "patient_name", "timestamp", "day_of_week",
    "risk_level", "risk_score",
    "ulcer_risk", "neuropathy_risk",
    "alert_triggered", "alert_type",
    "reading_quality", "medical_condition",
]


def load_artifacts():
    model = joblib.load(MODEL_DIR / "risk_model.pkl")
    needs_scale = joblib.load(MODEL_DIR / "needs_scaling.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, needs_scale, le, features


def plot_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("⚠️  Model has no native feature importance.")
        return

    n = min(top_n, len(feature_names))
    idx = np.argsort(importances)[-n:]

    plt.figure(figsize=(9, max(6, n * 0.35)))
    plt.barh(np.array(feature_names)[idx], importances[idx], color="teal")
    plt.title(f"Top {n} Feature Importances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=120)
    plt.close()
    print("✅ Feature importance saved.")


def run_shap(model, X_sample, feature_names, le):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        def save_plot(fig_path, plot_fn, *args, **kwargs):
            plt.figure(figsize=(10, 6))
            plot_fn(*args, **kwargs)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=120)
            plt.close()

        if isinstance(shap_values, list):
            for i, cls in enumerate(le.classes_):
                class_values = shap_values[i]
                try:
                    save_plot(
                        OUT_DIR / f"shap_{cls}.png",
                        shap.summary_plot,
                        class_values,
                        X_sample,
                        feature_names=feature_names,
                        show=False,
                        max_display=10,
                    )
                except TypeError:
                    save_plot(
                        OUT_DIR / f"shap_{cls}.png",
                        shap.plots.beeswarm,
                        class_values,
                        max_display=10,
                    )
            print("✅ SHAP per-class plots saved.")
        else:
            try:
                save_plot(
                    OUT_DIR / "shap_summary.png",
                    shap.summary_plot,
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                    max_display=15,
                )
            except TypeError:
                save_plot(
                    OUT_DIR / "shap_summary.png",
                    shap.plots.beeswarm,
                    shap_values,
                    max_display=15,
                )
            print("✅ SHAP summary saved.")
    except Exception as e:
        print(f"⚠️  SHAP skipped: {e}")


if __name__ == "__main__":
    model, needs_scale, le, features = load_artifacts()

    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=features, fill_value=0)

    plot_feature_importance(model, features)

    if hasattr(model, "feature_importances_"):
        X_sample = X.sample(min(500, len(X)), random_state=42)
        run_shap(model, X_sample, features, le)
    else:
        print("ℹ️  Skipping SHAP for non-tree model.")