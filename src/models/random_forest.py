"""M04 / Duc Anh: Random Forest baseline, CV tuning and pipeline export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.data_split import ROOT, load_config, load_train_data
from src.evaluation import (
    build_metrics_record,
    evaluate_cv,
    get_continuous_scores,
    save_metrics_json,
)
from src.preprocessing import build_pipeline, get_cv_splitter

MODEL_NAME = "random_forest"

# Search space for T23: 4 required Random Forest hyperparameters with `model__` prefix
# matching the M03 pipeline step name ("model").
PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [8, 12, 16, 20, None],
    "model__min_samples_leaf": [1, 5, 10, 20, 40],
    "model__max_features": ["sqrt", "log2", 0.5],
}

# Best configuration discovered by RandomizedSearchCV(n_iter=20, random_state=42) on 5-fold CV.
BEST_PARAMS: dict[str, Any] = {
    "model__n_estimators": 300,
    "model__max_depth": 16,
    "model__min_samples_leaf": 20,
    "model__max_features": "sqrt",
}


def build_baseline_pipeline(
    X_train: pd.DataFrame,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the default RandomForestClassifier wrapped in the shared M03 pipeline."""
    estimator = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return build_pipeline(estimator, X_train, model_type=MODEL_NAME)


def build_tuned_pipeline(
    X_train: pd.DataFrame,
    best_params: dict[str, Any] | None = None,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the tuned Random Forest pipeline by applying best_params onto the M03 pipeline."""
    pipeline = build_baseline_pipeline(X_train, random_state=random_state, n_jobs=n_jobs)
    params = best_params if best_params is not None else BEST_PARAMS
    pipeline.set_params(**params)
    return pipeline


def run_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Evaluate the default Random Forest baseline with 5-fold StratifiedKFold CV."""
    if config is None:
        config = load_config()
    seed = config["split"]["random_state"]
    pipeline = build_baseline_pipeline(X_train, random_state=seed, n_jobs=n_jobs)
    # Parallelism lives inside RandomForestClassifier (n_jobs=-1), so outer CV uses n_jobs=1.
    result = evaluate_cv(pipeline, X_train, y_train, config=config, n_jobs=1)
    return {
        "pipeline": pipeline,
        "validation_cv": result["validation_cv"],
        "training_time_seconds": result["training_time_seconds"],
    }


def run_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    param_distributions: dict[str, list[Any]] | None = None,
    n_iter: int = 20,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Tune n_estimators, max_depth, min_samples_leaf, and max_features via 5-fold CV."""
    if config is None:
        config = load_config()
    seed = config["split"]["random_state"]
    primary_metric = config["training"]["primary_metric"]
    cv = get_cv_splitter(config)

    base_pipeline = build_baseline_pipeline(X_train, random_state=seed, n_jobs=n_jobs)
    search_space = param_distributions or PARAM_DISTRIBUTIONS

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=primary_metric,
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=False,
        return_train_score=False,
    )

    start_time = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    results_df = (
        pd.DataFrame(search.cv_results_)
        .loc[
            :,
            [
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "param_model__n_estimators",
                "param_model__max_depth",
                "param_model__min_samples_leaf",
                "param_model__max_features",
            ],
        ]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    return {
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "search_time_seconds": float(elapsed),
        "cv_results_df": results_df,
    }


def train_and_export(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    best_params: dict[str, Any] | None = None,
    baseline_ap_mean: float | None = None,
    config: dict[str, Any] | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Evaluate tuned RF on 5-fold CV, fit final pipeline on X_train, and export artifacts."""
    if config is None:
        config = load_config()
    seed = config["split"]["random_state"]
    params = dict(best_params if best_params is not None else BEST_PARAMS)

    if baseline_ap_mean is None:
        baseline = run_baseline(X_train, y_train, config=config, n_jobs=n_jobs)
        baseline_ap_mean = float(baseline["validation_cv"]["average_precision_mean"])

    tuned_pipeline = build_tuned_pipeline(
        X_train, best_params=params, random_state=seed, n_jobs=n_jobs
    )
    cv_result = evaluate_cv(tuned_pipeline, X_train, y_train, config=config, n_jobs=1)

    # Fit the final pipeline once on 100% of X_train (never touching sealed test set).
    fit_start = time.perf_counter()
    tuned_pipeline.fit(X_train, y_train)
    final_fit_seconds = time.perf_counter() - fit_start

    _, score_method = get_continuous_scores(tuned_pipeline, X_train.iloc[:5])

    artifact_rel_path = f"{config['paths']['artifacts']}/{MODEL_NAME}.joblib"
    artifact_path = ROOT / artifact_rel_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipeline, artifact_path, compress=3)

    metrics_path = ROOT / config["paths"]["metrics"] / f"{MODEL_NAME}.json"
    record = build_metrics_record(
        model_name=MODEL_NAME,
        validation_cv=cv_result["validation_cv"],
        best_params=params,
        training_time_seconds=cv_result["training_time_seconds"],
        baseline_ap_mean=baseline_ap_mean,
        pipeline_path=artifact_rel_path,
        score_method=score_method,
        config=config,
    )
    save_metrics_json(record, metrics_path, config=config)

    return {
        "pipeline": tuned_pipeline,
        "record": record,
        "validation_cv": cv_result["validation_cv"],
        "cv_time_seconds": cv_result["training_time_seconds"],
        "final_fit_seconds": float(final_fit_seconds),
        "artifact_path": artifact_path,
        "metrics_path": metrics_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Random Forest M04 experiments.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "tune", "export", "full"],
        default="export",
        help="Run baseline, hyperparameter tuning, final export, or full pipeline.",
    )
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations.")
    args = parser.parse_args()

    cfg = load_config()
    X_tr, y_tr = load_train_data(config=cfg)

    if args.mode == "baseline":
        baseline = run_baseline(X_tr, y_tr, config=cfg)
        print(f"Baseline training time (5-fold CV): {baseline['training_time_seconds']:.2f}s")
        for metric, value in baseline["validation_cv"].items():
            print(f"  {metric:24s}: {value:.4f}")
    elif args.mode == "tune":
        tuning = run_tuning(X_tr, y_tr, config=cfg, n_iter=args.n_iter)
        print(f"Tuning time ({args.n_iter} candidates x 5 folds): {tuning['search_time_seconds']:.2f}s")
        print(f"Best AP (5-fold CV): {tuning['best_score']:.4f}")
        print(f"Best params: {tuning['best_params']}")
        print("\nTop 10 candidates:")
        print(tuning["cv_results_df"].head(10).to_string(index=False))
    elif args.mode == "export":
        out = train_and_export(X_tr, y_tr, config=cfg)
        print(f"Tuned 5-fold CV time : {out['cv_time_seconds']:.2f}s")
        print(f"Final fit time       : {out['final_fit_seconds']:.2f}s")
        print(f"Saved pipeline       : {out['artifact_path']}")
        print(f"Saved metrics JSON   : {out['metrics_path']}")
        print("\nMetrics record:")
        print(json.dumps(out["record"], indent=2, ensure_ascii=False))
    elif args.mode == "full":
        baseline = run_baseline(X_tr, y_tr, config=cfg)
        tuning = run_tuning(X_tr, y_tr, config=cfg, n_iter=args.n_iter)
        out = train_and_export(
            X_tr,
            y_tr,
            best_params=tuning["best_params"],
            baseline_ap_mean=baseline["validation_cv"]["average_precision_mean"],
            config=cfg,
        )
        print(f"Saved pipeline       : {out['artifact_path']}")
        print(f"Saved metrics JSON   : {out['metrics_path']}")
        print(json.dumps(out["record"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
