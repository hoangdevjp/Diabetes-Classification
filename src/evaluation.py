"""M03 / Hien and Hoang: agree on shared metric definitions and output schema.

Report accuracy, precision(1), recall(1), F1(1), ROC-AUC and average precision.
Use continuous scores for ROC-AUC/AP; reserve final-test evaluation for M06.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.data_split import load_config
from src.preprocessing import get_cv_splitter

SCORING_METRICS: dict[str, Any] = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

REQUIRED_CV_KEYS: tuple[str, ...] = tuple(
    key
    for metric in SCORING_METRICS
    for key in (f"{metric}_mean", f"{metric}_std")
)

REQUIRED_RECORD_KEYS: tuple[str, ...] = (
    "model_name",
    "dataset_version",
    "target_column",
    "positive_label",
    "train_test_split",
    "cv",
    "selection_metric",
    "baseline",
    "best_params",
    "validation_cv",
    "training_time_seconds",
    "pipeline_path",
)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def get_continuous_scores(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    """Extract continuous positive-class scores via predict_proba or decision_function."""
    predict_proba = getattr(estimator, "predict_proba", None)
    if callable(predict_proba):
        proba = np.asarray(predict_proba(X), dtype=float)
        if proba.ndim == 2:
            pos_idx = 1
            classes = getattr(estimator, "classes_", None)
            if classes is not None and 1 in list(classes):
                pos_idx = list(classes).index(1)
            return proba[:, pos_idx], "predict_proba"
        return proba.ravel(), "predict_proba"

    decision_function = getattr(estimator, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(X), dtype=float)
        return scores.ravel(), "decision_function"

    raise TypeError(
        "Estimator must implement predict_proba or decision_function for ROC-AUC/AP; "
        "discrete predict() labels are not allowed."
    )


def compute_classification_metrics(
    y_true: Any, y_pred: Any, y_score: Any
) -> dict[str, float]:
    """Compute the 6 shared binary classification metrics on a single evaluation split."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def evaluate_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run 5-fold StratifiedKFold CV on X_train and aggregate mean/std (ddof=0) metrics."""
    if config is None:
        config = load_config()
    cv = get_cv_splitter(config)

    start_time = time.perf_counter()
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRICS,
        return_train_score=False,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - start_time

    summary: dict[str, float] = {}
    for metric_name in SCORING_METRICS:
        scores = np.asarray(cv_results[f"test_{metric_name}"], dtype=float)
        summary[f"{metric_name}_mean"] = float(np.mean(scores))
        summary[f"{metric_name}_std"] = float(np.std(scores, ddof=0))

    return {
        "validation_cv": summary,
        "training_time_seconds": float(elapsed),
    }


def build_metrics_record(
    model_name: str,
    validation_cv: dict[str, float],
    best_params: dict[str, Any],
    training_time_seconds: float,
    *,
    baseline_ap_mean: float | None = None,
    pipeline_path: str | None = None,
    score_method: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metrics.json payload conforming to the shared project contract."""
    if config is None:
        config = load_config()

    dataset_cfg = config["dataset"]
    cv_cfg = config["cv"]
    record: dict[str, Any] = {
        "model_name": model_name,
        "dataset_version": dataset_cfg.get("version", "TO_CONFIRM"),
        "target_column": dataset_cfg["target_column"],
        "positive_label": dataset_cfg["positive_label"],
        "train_test_split": dict(config["split"]),
        "cv": {
            "strategy": cv_cfg["strategy"],
            "folds": cv_cfg["n_splits"],
            "shuffle": cv_cfg["shuffle"],
            "random_state": cv_cfg["random_state"],
        },
        "selection_metric": config["training"]["primary_metric"],
        "baseline": {
            "average_precision_mean": (
                None if baseline_ap_mean is None else float(baseline_ap_mean)
            )
        },
        "best_params": dict(best_params),
        "validation_cv": {key: float(validation_cv[key]) for key in REQUIRED_CV_KEYS if key in validation_cv},
        "training_time_seconds": float(training_time_seconds),
        "pipeline_path": pipeline_path or f"artifacts/{model_name}.joblib",
    }
    if score_method is not None:
        record["score_method"] = score_method
    return record


def validate_metrics_record(
    record: dict[str, Any], config: dict[str, Any] | None = None
) -> None:
    """Validate metrics.json structure, config consistency, and finite metric values."""
    if config is None:
        config = load_config()

    missing_top = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing_top:
        raise ValueError(f"metrics.json missing required keys: {missing_top}")

    if not isinstance(record["model_name"], str) or not record["model_name"].strip():
        raise ValueError("'model_name' must be a non-empty string.")
    if not isinstance(record["best_params"], dict):
        raise ValueError("'best_params' must be a dictionary.")
    if not isinstance(record["baseline"], dict) or "average_precision_mean" not in record["baseline"]:
        raise ValueError("'baseline' must contain 'average_precision_mean'.")
    baseline_ap = record["baseline"]["average_precision_mean"]
    if baseline_ap is not None and not _is_finite_number(baseline_ap):
        raise ValueError("'baseline.average_precision_mean' must be null or a finite number.")

    if not _is_finite_number(record["training_time_seconds"]) or float(record["training_time_seconds"]) < 0:
        raise ValueError("'training_time_seconds' must be a finite non-negative number.")

    expected_primary = config["training"]["primary_metric"]
    if record["selection_metric"] != expected_primary:
        raise ValueError(
            f"'selection_metric' mismatch: expected '{expected_primary}', got '{record['selection_metric']}'."
        )
    if record["target_column"] != config["dataset"]["target_column"]:
        raise ValueError("Target column in metrics record does not match experiment.yaml.")
    if record["positive_label"] != config["dataset"]["positive_label"]:
        raise ValueError("Positive label in metrics record does not match experiment.yaml.")
    if record["train_test_split"] != dict(config["split"]):
        raise ValueError("train_test_split in metrics record does not match experiment.yaml.")

    cv_cfg = config["cv"]
    expected_cv = {
        "strategy": cv_cfg["strategy"],
        "folds": cv_cfg["n_splits"],
        "shuffle": cv_cfg["shuffle"],
        "random_state": cv_cfg["random_state"],
    }
    if record["cv"] != expected_cv:
        raise ValueError(f"cv config mismatch: expected {expected_cv}, got {record['cv']}.")

    val_cv = record["validation_cv"]
    if not isinstance(val_cv, dict):
        raise ValueError("'validation_cv' must be a dictionary.")
    for key in REQUIRED_CV_KEYS:
        if key not in val_cv:
            raise ValueError(f"validation_cv missing required key: '{key}'")
        if not _is_finite_number(val_cv[key]):
            raise ValueError(f"validation_cv['{key}'] must be a finite number, got {val_cv[key]!r}.")


def save_metrics_json(
    record: dict[str, Any],
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Validate and serialize a metrics record to UTF-8 JSON."""
    validate_metrics_record(record, config=config)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return destination
