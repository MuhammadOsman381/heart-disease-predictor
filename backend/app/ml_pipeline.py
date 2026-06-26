from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.storage import get_storage


CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
RAW_COLUMNS = CONTINUOUS_COLS + CATEGORICAL_COLS + ["target"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def train_from_csv(csv_path: Path, run_id: UUID) -> dict[str, Any]:
    settings = get_settings()
    storage = get_storage()
    run_dir = settings.artifact_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_columns = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    df = df[RAW_COLUMNS].copy()
    rows_before = int(len(df))
    duplicate_count = int(df.duplicated().sum())
    missing_values = df.isna().sum().to_dict()
    target_distribution_before = df["target"].value_counts().to_dict()

    df = df.drop_duplicates().dropna().reset_index(drop=True)
    rows_after = int(len(df))

    X_raw = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=False)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    feature_columns = X_train.columns.tolist()
    feature_engineering_stats = {
        "engineered_features": ["risk_stratification_index", "st_segment_strain_index"],
        "feature_count": len(feature_columns),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "class_balance_train": y_train.value_counts(normalize=True).round(4).to_dict(),
        "class_balance_test": y_test.value_counts(normalize=True).round(4).to_dict(),
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }

    baseline_results: dict[str, dict[str, float]] = {}
    for name, model in baseline_models.items():
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="recall", n_jobs=1)
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=1)
        cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="roc_auc", n_jobs=1)
        baseline_results[name] = {
            "mean_recall": float(cv_recall.mean()),
            "std_recall": float(cv_recall.std()),
            "mean_accuracy": float(cv_accuracy.mean()),
            "mean_roc_auc": float(cv_roc_auc.mean()),
        }

    weights = {"mean_recall": 0.5, "mean_accuracy": 0.2, "mean_roc_auc": 0.3}
    for result in baseline_results.values():
        result["weighted_score"] = sum(result[k] * weight for k, weight in weights.items())

    champion_model = max(baseline_results, key=lambda name: baseline_results[name]["weighted_score"])
    if champion_model == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    evaluation_stats = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Disease"],
            output_dict=True,
            zero_division=0,
        ),
    }

    threshold_df = []
    for threshold in np.arange(0.05, 0.95, 0.01):
        adjusted = (y_pred_proba >= threshold).astype(int)
        threshold_df.append(
            {
                "threshold": round(float(threshold), 2),
                "recall": float(recall_score(y_test, adjusted, zero_division=0)),
                "precision": float(precision_score(y_test, adjusted, zero_division=0)),
                "f1": float(f1_score(y_test, adjusted, zero_division=0)),
            }
        )

    candidates = [row for row in threshold_df if row["recall"] >= 0.97]
    optimal_row = max(candidates, key=lambda row: row["threshold"]) if candidates else max(threshold_df, key=lambda row: row["recall"])
    optimal_threshold = float(optimal_row["threshold"])
    evaluation_stats["optimal_threshold_result"] = optimal_row

    paths = {
        "model": run_dir / "heart_model.pkl",
        "scaler": run_dir / "scaler.pkl",
        "feature_columns": run_dir / "feature_columns.pkl",
        "threshold": run_dir / "optimal_threshold.pkl",
    }
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_columns, paths["feature_columns"])
    joblib.dump(optimal_threshold, paths["threshold"])

    remote_paths = {
        name: storage.upload_file(local_path, f"{run_id}/{local_path.name}")
        for name, local_path in paths.items()
    }

    preprocessing_stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicate_count,
        "missing_values_before_drop": missing_values,
        "target_distribution_before": target_distribution_before,
        "target_distribution_after": df["target"].value_counts().to_dict(),
        "continuous_columns_scaled": CONTINUOUS_COLS,
        "categorical_columns_encoded": CATEGORICAL_COLS,
    }

    model_selection_stats = {
        "baseline_results": baseline_results,
        "weights": weights,
        "champion_model": champion_model,
        "final_model_params": model.get_params(),
    }

    return _json_safe(
        {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "feature_count": len(feature_columns),
            "champion_model": champion_model,
            "optimal_threshold": optimal_threshold,
            "preprocessing_stats": preprocessing_stats,
            "feature_engineering_stats": feature_engineering_stats,
            "model_selection_stats": model_selection_stats,
            "evaluation_stats": evaluation_stats,
            "model_artifact_path": remote_paths["model"],
            "scaler_artifact_path": remote_paths["scaler"],
            "feature_columns_artifact_path": remote_paths["feature_columns"],
            "threshold_artifact_path": remote_paths["threshold"],
        }
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["risk_stratification_index"] = (
        result["age"] * 0.3
        + result["trestbps"] * 0.35
        + result["chol"] * 0.35
    ) + (result["age"] * result["trestbps"] * result["chol"] * 0.05)

    slope_weights = {"slope_0": 0.5, "slope_1": 1.0, "slope_2": 1.5}
    result["slope_severity_weight"] = 0.0
    for column, weight in slope_weights.items():
        if column in result.columns:
            result["slope_severity_weight"] += result[column] * weight

    result["st_segment_strain_index"] = result["oldpeak"] * result["slope_severity_weight"]
    return result.drop(columns=["slope_severity_weight"])


def predict_patient(patient: dict[str, Any], artifacts: dict[str, Path]) -> dict[str, Any]:
    model = joblib.load(artifacts["model"])
    scaler = joblib.load(artifacts["scaler"])
    feature_columns = joblib.load(artifacts["feature_columns"])
    threshold = float(joblib.load(artifacts["threshold"]))

    row = pd.DataFrame([{key: patient[key] for key in CONTINUOUS_COLS + CATEGORICAL_COLS}])
    row[CONTINUOUS_COLS] = scaler.transform(row[CONTINUOUS_COLS])
    row = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=False)
    row = add_engineered_features(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    probability = float(model.predict_proba(row)[:, 1][0])
    prediction = int(probability >= threshold)
    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "label": "Disease risk" if prediction else "No disease risk",
    }
