"""
VedaSync - Step 5: Evaluate and Plot Results
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_comparison(metrics):
    rows = []
    for name, res in metrics["results"].items():
        rows.append({
            "model":       name,
            "accuracy":    res["accuracy"],
            "weighted_f1": res["weighted_f1"],
            "roc_auc":     res["roc_auc"] or 0.0,
        })
    df = pd.DataFrame(rows).set_index("model")

    df.plot(kind="bar", figsize=(10, 5), colormap="viridis")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=120)
    plt.close()
    print("✅ Model comparison plot saved.")


def plot_confusion_matrices(metrics, label_encoder_path=None):
    label_encoder_path = Path(label_encoder_path) if label_encoder_path else BASE_DIR / "models" / "label_encoder.pkl"
    try:
        le = joblib.load(label_encoder_path)
        labels = [str(c) for c in le.classes_]
    except Exception:
        labels = None

    for name, res in metrics["results"].items():
        cm = np.array(res["confusion_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_{name}.png", dpi=120)
        plt.close()
    print("✅ Confusion matrices saved.")


if __name__ == "__main__":
    with open(OUT_DIR / "model_metrics.json") as f:
        metrics = json.load(f)

    plot_model_comparison(metrics)
    plot_confusion_matrices(metrics)

    print(f"\n🏆 Best Model: {metrics['best_model']}")
    print(f"📊 CV F1: {np.mean(metrics['cv_f1']):.4f}")