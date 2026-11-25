"""M04 / Hung: Linear/RBF SVM experiments, CV tuning and pipeline export.

Decisions:
  - Linear SVM  : LinearSVC  (decision_function, faster on 183k rows)
  - RBF SVM     : SVC(kernel='rbf', probability=False)  (decision_function)
  - class_weight: 'balanced' as baseline (positive_rate ~15.3%)
  - GridSearchCV: AP as scoring; 10 points Linear, 9 points RBF
  - Artifacts   : artifacts/linear_svm.joblib, artifacts/rbf_svm.joblib
  - Metrics     : results/metrics/linear_svm.json, results/metrics/rbf_svm.json
  - Pipeline key: ("preprocessor", "model")  — grid prefix: model__<param>
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.data_split import load_config
from src.evaluation import build_metrics_record, evaluate_cv, save_metrics_json
from src.preprocessing import build_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

LINEAR_SVM_DEFAULTS: dict[str, Any] = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42,
    "dual": "auto",   # sklearn >= 1.3: auto selects dual/primal based on n_samples vs n_features
}

RBF_SVM_DEFAULTS: dict[str, Any] = {
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "random_state": 42,
}

# GridSearch spaces — tuning on train folds only via GridSearchCV
LINEAR_PARAM_GRID: dict[str, list[Any]] = {
    "model__C": [0.001, 0.01, 0.1, 1, 10],
    "model__class_weight": ["balanced", None],
}

RBF_PARAM_GRID: dict[str, list[Any]] = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto", 0.01],
    "model__class_weight": ["balanced"],
}


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_linear_svm_pipeline(
    X: pd.DataFrame,
    *,
    C: float = LINEAR_SVM_DEFAULTS["C"],
    class_weight: Any = LINEAR_SVM_DEFAULTS["class_weight"],
    max_iter: int = LINEAR_SVM_DEFAULTS["max_iter"],
    random_state: int = LINEAR_SVM_DEFAULTS["random_state"],
    dual: Any = LINEAR_SVM_DEFAULTS["dual"],
) -> Pipeline:
    """Return an unfitted sklearn Pipeline: StandardScaler → LinearSVC.

    model_type='linear_svm' ensures StandardScaler is applied (required for SVM).
    Pipeline step names: ('preprocessor', 'model') — grid prefix: model__<param>.
    """
    estimator = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        dual=dual,
    )
    return build_pipeline(estimator, X, model_type="linear_svm")


def build_rbf_svm_pipeline(
    X: pd.DataFrame,
    *,
    C: float = RBF_SVM_DEFAULTS["C"],
    gamma: Any = RBF_SVM_DEFAULTS["gamma"],
    class_weight: Any = RBF_SVM_DEFAULTS["class_weight"],
    random_state: int = RBF_SVM_DEFAULTS["random_state"],
) -> Pipeline:
    """Return an unfitted sklearn Pipeline: StandardScaler → SVC(kernel='rbf').

    probability=False for speed (uses decision_function, consistent with LinearSVC).
    model_type='rbf_svm' ensures StandardScaler is applied.
    """
    estimator = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        probability=False,
    )
    return build_pipeline(estimator, X, model_type="rbf_svm")


# ---------------------------------------------------------------------------
# Baseline CV (before tuning)
# ---------------------------------------------------------------------------


def run_baseline_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run 5-fold StratifiedKFold CV on the baseline pipeline and return summary dict.

    Returns the dict produced by evaluate_cv():
        {"validation_cv": {metric_mean, metric_std, ...}, "training_time_seconds": float}
    """
    if config is None:
        config = load_config()
    return evaluate_cv(pipeline, X_train, y_train, config=config, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def run_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list[Any]],
    *,
    config: dict[str, Any] | None = None,
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Run GridSearchCV with 5-fold StratifiedKFold and scoring='average_precision'.

    Returns a *fitted* GridSearchCV object (best_estimator_ accessible via .best_estimator_).
    All fitting happens strictly on X_train; no test data is involved.
    """
    from src.preprocessing import get_cv_splitter  # avoid circular at module level

    if config is None:
        config = load_config()
    cv = get_cv_splitter(config)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=False,
    )
    start = time.perf_counter()
    gs.fit(X_train, y_train)
    gs.search_time_seconds_ = time.perf_counter() - start  # type: ignore[attr-defined]
    return gs


# ---------------------------------------------------------------------------
# Save and load helpers
# ---------------------------------------------------------------------------


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline to disk using joblib and return the saved path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    return dest


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    return joblib.load(Path(path))


# ---------------------------------------------------------------------------
# High-level experiment runner (optional — used by notebook)
# ---------------------------------------------------------------------------


def run_svm_experiment(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    artifacts_dir: Path | None = None,
    metrics_dir: Path | None = None,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict[str, Any]:
    """End-to-end experiment: baseline CV → GridSearch → save pipeline + metrics JSON.

    Args:
        model_name: Identifier used for artifact names (e.g. 'linear_svm', 'rbf_svm').
        pipeline: Unfitted pipeline built by build_linear_svm_pipeline / build_rbf_svm_pipeline.
        param_grid: GridSearchCV parameter grid.
        X_train: Training features (from load_train_data()).
        y_train: Training labels.
        config: Optional pre-loaded config dict.
        artifacts_dir: Override for artifacts output directory.
        metrics_dir: Override for metrics output directory.
        n_jobs: Parallelism for CV and GridSearchCV.
        verbose: GridSearchCV verbosity.

    Returns:
        A dict with keys: 'baseline_cv', 'gs', 'metrics_record', 'artifact_path', 'metrics_path'.
    """
    if config is None:
        config = load_config()

    # Resolve output paths from config
    _root = Path(__file__).resolve().parents[2]
    if artifacts_dir is None:
        artifacts_dir = _root / config["paths"]["artifacts"]
    if metrics_dir is None:
        metrics_dir = _root / config["paths"]["metrics"]

    artifact_path = artifacts_dir / f"{model_name}.joblib"
    metrics_path = metrics_dir / f"{model_name}.json"

    # --- Step 1: Baseline CV ---
    print(f"\n{'='*60}")
    print(f"[{model_name}] Step 1/3 — Baseline 5-fold CV")
    print("="*60)
    baseline_cv = run_baseline_cv(pipeline, X_train, y_train, config=config, n_jobs=n_jobs)
    baseline_ap = baseline_cv["validation_cv"]["average_precision_mean"]
    print(f"  Baseline AP (mean): {baseline_ap:.4f}")

    # --- Step 2: GridSearchCV tuning ---
    print(f"\n[{model_name}] Step 2/3 — GridSearchCV hyperparameter tuning")
    print(f"  Grid: {param_grid}")
    gs = run_grid_search(
        pipeline, X_train, y_train, param_grid,
        config=config, n_jobs=n_jobs, verbose=verbose,
    )
    best_ap = gs.best_score_
    best_params = gs.best_params_
    search_time = gs.search_time_seconds_
    print(f"  Best AP (CV): {best_ap:.4f}")
    print(f"  Best params: {best_params}")
    print(f"  Search time: {search_time:.1f}s")

    # --- Step 3: Evaluate best pipeline with full 6-metric CV ---
    print(f"\n[{model_name}] Step 3/3 — Evaluate best pipeline (full metrics CV)")
    best_pipeline_cv = run_baseline_cv(
        gs.best_estimator_, X_train, y_train, config=config, n_jobs=n_jobs
    )
    val_cv = best_pipeline_cv["validation_cv"]
    training_time = best_pipeline_cv["training_time_seconds"]

    # Save fitted best pipeline
    save_pipeline(gs.best_estimator_, artifact_path)
    print(f"  Pipeline saved → {artifact_path}")

    # Build and save metrics record
    # Determine score method from best estimator's inner model
    inner_model = gs.best_estimator_.named_steps["model"]
    score_method = (
        "predict_proba" if hasattr(inner_model, "predict_proba")
        and callable(inner_model.predict_proba)
        else "decision_function"
    )

    metrics_record = build_metrics_record(
        model_name=model_name,
        validation_cv=val_cv,
        best_params=best_params,
        training_time_seconds=training_time,
        baseline_ap_mean=baseline_ap,
        pipeline_path=str(artifact_path.relative_to(_root)).replace("\\", "/"),
        score_method=score_method,
        config=config,
    )
    saved_metrics = save_metrics_json(metrics_record, metrics_path, config=config)
    print(f"  Metrics saved → {saved_metrics}")

    return {
        "baseline_cv": baseline_cv,
        "gs": gs,
        "metrics_record": metrics_record,
        "artifact_path": artifact_path,
        "metrics_path": metrics_path,
    }

