"""M03 / Hien: define the shared preprocessing API before model training.

Fit imputation, encoding and scaling only inside each training-fold pipeline.
M04 tuning grids target estimator parameters using the ``model__<param>`` prefix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_split import load_config

MODEL_SCALING_DEFAULTS: dict[str, bool] = {
    "naive_bayes": False,
    "gaussian_nb": False,
    "knn": True,
    "random_forest": False,
    "svm": True,
    "linear_svm": True,
    "rbf_svm": True,
}


def get_cv_splitter(config: dict[str, Any] | None = None) -> StratifiedKFold:
    """Return the shared 5-fold StratifiedKFold splitter configured in experiment.yaml."""
    if config is None:
        config = load_config()
    cv_cfg = config["cv"]
    if cv_cfg.get("strategy") != "StratifiedKFold":
        raise ValueError(f"Unsupported CV strategy: {cv_cfg.get('strategy')}")
    return StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )


def build_preprocessor(
    X: pd.DataFrame,
    model_type: str = "knn",
    *,
    scale_numeric: bool | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer tailored to the target model family.

    Numeric features receive median imputation and optional StandardScaler.
    Categorical features (if any) receive most-frequent imputation and OneHotEncoder.
    """
    model_key = model_type.lower().strip()
    if scale_numeric is None:
        if model_key not in MODEL_SCALING_DEFAULTS:
            valid = sorted(MODEL_SCALING_DEFAULTS)
            raise ValueError(f"Unknown model_type '{model_type}'. Expected one of: {valid}")
        scale_numeric = MODEL_SCALING_DEFAULTS[model_key]

    if numeric_cols is None and categorical_cols is None:
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]
    elif numeric_cols is None:
        cat_set = set(categorical_cols or [])
        numeric_cols = [col for col in X.columns if col not in cat_set]
    elif categorical_cols is None:
        num_set = set(numeric_cols)
        categorical_cols = [col for col in X.columns if col not in num_set]

    if not numeric_cols and not categorical_cols:
        raise ValueError("Cannot build preprocessor without numeric or categorical columns.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), list(numeric_cols)))

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, list(categorical_cols)))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    model_type: str,
    *,
    scale_numeric: bool | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Wrap preprocessing and estimator into a leak-free sklearn Pipeline."""
    preprocessor = build_preprocessor(
        X,
        model_type=model_type,
        scale_numeric=scale_numeric,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
