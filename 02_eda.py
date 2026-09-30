"""
VedaSync - Step 2: Exploratory Data Analysis
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "veda_sync_clean.csv"
OUT_DIR = BASE_DIR / "outputs" / "eda_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")


def run_eda(df: pd.DataFrame):
    # ---------- 1. Target distributions ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df["risk_level"].value_counts().plot(kind="bar", ax=ax[0], color="steelblue")
    ax[0].set_title("Risk Level Distribution")
    ax[0].set_ylabel("Count")

    df["alert_triggered"].value_counts().plot(kind="bar", ax=ax[1], color="tomato")
    ax[1].set_title("Alert Triggered")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/target_distributions.png", dpi=120)
    plt.close()

    # ---------- 2. Sensor distributions by risk level ----------
    sensors = ["temperature_celsius", "pressure_grams",
               "moisture_rh_percent", "spo2_percent"]
    sensors = [s for s in sensors if s in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), sensors):
        sns.boxplot(
            data=df, x="risk_level", y=col,
            hue="risk_level", dodge=False,
            palette="Set2", legend=False, ax=ax
        )
        ax.set_title(f"{col} by Risk Level")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sensor_distributions.png", dpi=120)
    plt.close()

    # ---------- 3. Correlation heatmap ----------
    numeric = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()

    # ---------- 4. Temperature asymmetry ----------
    df_pivot = df.pivot_table(
        index=["patient_id", "timestamp"],
        columns="sensor_location",
        values="temperature_celsius",
        aggfunc="mean"
    ).reset_index()

    if "Heel" in df_pivot.columns and "Big_Toe" in df_pivot.columns:
        df_pivot["temp_asymmetry"] = (
            df_pivot["Heel"] - df_pivot["Big_Toe"]
        ).abs()
        plt.figure(figsize=(8, 4))
        sns.histplot(df_pivot["temp_asymmetry"].dropna(), bins=30, kde=True)
        plt.title("Heel ↔ Big Toe Temperature Asymmetry")
        plt.xlabel("|ΔT| °C")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/temp_asymmetry.png", dpi=120)
        plt.close()
        print("✅ Temperature asymmetry plot saved.")
    else:
        print("⚠️  Heel or Big_Toe missing — skipping asymmetry plot.")

    # ---------- 5. Alert type distribution ----------
    alerts = df[df["alert_triggered"] == "Yes"]
    if not alerts.empty:
        plt.figure(figsize=(8, 4))
        alerts["alert_type"].value_counts().plot(kind="bar", color="darkorange")
        plt.title("Alert Type Distribution")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/alert_types.png", dpi=120)
        plt.close()
    else:
        print("⚠️  No alerts triggered — skipping alert type plot.")

    print("\n📊 Numerical summary:")
    print(df.describe().T)
    print("\n📊 Risk level share:")
    print((df["risk_level"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    run_eda(df)
    print(f"\n✅ All EDA plots saved to {OUT_DIR}/")