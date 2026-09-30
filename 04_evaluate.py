"""Report held-out risk performance, warning lead time, and SHAP importance."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "risk_model_bundle.joblib"
PREDICTIONS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.json"
OUT_DIR = BASE_DIR / "outputs"


def evaluate_warnings(predictions: pd.DataFrame, horizon_days: int = 7) -> dict:
    if "simulated_ulcer_event_time" not in predictions:
        return {"warning_recall": None, "warning_precision": None, "mean_lead_time_hours": None,
                "note": "No event timestamps available; lead-time metrics are not estimable."}

    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["simulated_ulcer_event_time"] = pd.to_datetime(
        frame["simulated_ulcer_event_time"], errors="coerce"
    )
    frame["warning"] = frame["predicted_risk_level"].astype(str).str.lower().eq("high")
    has_event = frame["simulated_ulcer_event_time"].notna()
    frame["event_in_horizon"] = (
        has_event
        & frame["timestamp"].lt(frame["simulated_ulcer_event_time"])
        & frame["timestamp"].ge(frame["simulated_ulcer_event_time"] - pd.Timedelta(days=horizon_days))
    )

    device_rows = []
    for device_id, group in frame.groupby("device_id"):
        event_time = group["simulated_ulcer_event_time"].dropna()
        has_event_for_device = not event_time.empty
        warning_rows = group[group["warning"] & group["event_in_horizon"]]
        detected = not warning_rows.empty
        lead_hours = None
        if detected and has_event_for_device:
            first_warning = warning_rows["timestamp"].min()
            lead_hours = float((event_time.min() - first_warning).total_seconds() / 3600)
        device_rows.append({
            "device_id": device_id,
            "event": has_event_for_device,
            "warning": bool(warning_rows.shape[0]) if has_event_for_device else bool(group["warning"].any()),
            "detected_in_horizon": detected,
            "lead_time_hours": lead_hours,
        })
    devices = pd.DataFrame(device_rows)
    events = devices["event"]
    warnings = devices["warning"]
    detected_events = devices["detected_in_horizon"]
    lead_times = devices["lead_time_hours"].dropna()
    return {
        "event_devices": int(events.sum()),
        "warning_devices": int(warnings.sum()),
        "detected_event_devices_within_7d": int(detected_events.sum()),
        "warning_recall": float(detected_events.sum() / events.sum()) if events.sum() else None,
        "warning_precision": float((warnings & events).sum() / warnings.sum()) if warnings.sum() else None,
        "mean_lead_time_hours": float(lead_times.mean()) if not lead_times.empty else None,
        "lead_time_horizon_days": horizon_days,
        "unit": "device, not sensor row",
    }


def compute_shap_importance(bundle: dict, sample: pd.DataFrame) -> pd.DataFrame:
    import shap

    model = bundle["model"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    model_input = imputer.transform(sample[bundle["raw_feature_names"]])
    if bundle["needs_scaling"]:
        model_input = scaler.transform(model_input)
    feature_names = bundle["model_feature_names"]

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input[: min(80, len(model_input))], check_additivity=False)
    else:
        explainer = shap.LinearExplainer(model, model_input[: min(80, len(model_input))])
        shap_values = explainer.shap_values(model_input[: min(80, len(model_input))])

    if isinstance(shap_values, list):
        values = np.stack([np.abs(class_values) for class_values in shap_values]).mean(axis=0)
    else:
        values = np.abs(np.asarray(shap_values))
        if values.ndim == 3:
            values = values.mean(axis=-1)
    importance = values.mean(axis=0)
    result = pd.DataFrame({"feature": feature_names, "mean_absolute_shap": importance})
    return result.sort_values("mean_absolute_shap", ascending=False)


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(MODEL_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    features = pd.read_csv(BASE_DIR / "data" / "veda_sync_demo_features.csv")
    warnings = evaluate_warnings(predictions)
    metrics["warning_metrics"] = warnings

    truth = predictions["risk_level"].astype(str)
    predicted = predictions["predicted_risk_level"].astype(str)
    metrics["selected_model_classification_report"] = classification_report(
        truth, predicted, labels=bundle["label_encoder"].classes_.tolist(),
        output_dict=True, zero_division=0,
    )
    metrics["selected_model_precision_macro"] = float(precision_score(truth, predicted, average="macro", zero_division=0))
    metrics["selected_model_recall_macro"] = float(recall_score(truth, predicted, average="macro", zero_division=0))

    split_devices = set(predictions["device_id"])
    sample = features[features["device_id"].isin(split_devices)]
    try:
        importance = compute_shap_importance(bundle, sample)
        importance.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)
        metrics["shap_top_features"] = importance.head(15).to_dict(orient="records")
        metrics["shap_status"] = "computed"
    except Exception as error:
        metrics["shap_status"] = f"unavailable: {type(error).__name__}: {error}"
        print(f"SHAP calculation unavailable: {error}")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": metrics["best_model"], "warning_metrics": warnings,
                      "shap_status": metrics["shap_status"]}, indent=2))


if __name__ == "__main__":
    main()
