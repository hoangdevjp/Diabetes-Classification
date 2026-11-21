"""M03 / Hien: implement one stratified 80/20 split with random_state=42.

Keep test data for final evaluation in M06; record dataset hash and row indices.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
ROW_INDEX_NAME = "row_index"

# Phan 1: Doc config va tinh hash
def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the shared experiment configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 1 MB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash over column schema, index, and row values."""
    digest = hashlib.sha256()
    schema = [(str(col), str(dtype)) for col, dtype in zip(df.columns, df.dtypes)]
    digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()

def _hash_index(index: pd.Index) -> str:
    payload = ",".join(map(str, index)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# Phan 2: Kiem tra Schema
def validate_schema(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate target/feature schema and return (X, y) in canonical column order."""
    if config is None:
        config = load_config()
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if df.columns.has_duplicates:
        raise ValueError("Input DataFrame contains duplicate column names.")

    dataset_cfg = config["dataset"]
    target_col = dataset_cfg["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'.")

    y_raw = df[target_col]
    if y_raw.isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing value.")
    if pd.api.types.is_bool_dtype(y_raw) or not pd.api.types.is_numeric_dtype(y_raw):
        raise ValueError(f"Target column '{target_col}' must be integer/numeric 0/1.")

    unique_labels = set(y_raw.unique().tolist())
    if unique_labels != {0, 1}:
        raise ValueError(
            f"Target column '{target_col}' must contain binary classes {{0, 1}}, got {sorted(unique_labels)}."
        )

    configured_features = dataset_cfg.get("feature_columns")
    raw_feature_cols = [col for col in df.columns if col != target_col]
    if configured_features is not None:
        expected_set = set(configured_features)
        actual_set = set(raw_feature_cols)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        if missing or unexpected:
            raise ValueError(
                f"Feature schema mismatch (missing={missing}, unexpected={unexpected})."
            )
        feature_cols = list(configured_features)
    else:
        if "ID" in raw_feature_cols:
            raise ValueError("Non-predictor column 'ID' must be dropped before splitting.")
        feature_cols = raw_feature_cols

    if not feature_cols:
        raise ValueError("No feature columns remain after removing target.")

    X = df[feature_cols].copy()
    y = y_raw.astype("int64").copy()
    return X, y

# Phan 3: Chia 80/20 cho train/test
def split_train_test(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cleaned data into stratified 80% train and 20% test sets with seed 42."""
    if config is None:
        config = load_config()
    X, y = validate_schema(df, config)
    split_cfg = config["split"]
    stratify_y = y if split_cfg.get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        stratify=stratify_y,
        random_state=split_cfg["random_state"],
    )
    if not X_train.index.intersection(X_test.index).empty:
        raise RuntimeError("Train and test splits share overlapping indices.")
    return X_train, X_test, y_train, y_test
def _label_summary(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}

# Phan 4: Luu tap split & metadata
def save_splits(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Create and persist train.csv, test.csv, and split_metadata.json."""
    if config is None:
        config = load_config()
    splits_dir = Path(output_dir) if output_dir is not None else ROOT / config["paths"]["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_train_test(df, config)
    target_col = config["dataset"]["target_column"]
    train_df = X_train.assign(**{target_col: y_train})
    test_df = X_test.assign(**{target_col: y_test})
    train_df.to_csv(splits_dir / "train.csv", index=True, index_label=ROW_INDEX_NAME)
    test_df.to_csv(splits_dir / "test.csv", index=True, index_label=ROW_INDEX_NAME)
    source_sha = sha256_file(source_path) if source_path and Path(source_path).exists() else None
    metadata: dict[str, Any] = {
        "source_path": _display_path(Path(source_path)) if source_path is not None else None,
        "source_file_sha256": source_sha,
        "dataframe_sha256": compute_dataframe_hash(df),
        "target_column": target_col,
        "feature_columns": list(X_train.columns),
        "split_config": dict(config["split"]),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_label_counts": _label_summary(y_train),
        "test_label_counts": _label_summary(y_test),
        "train_positive_rate": float((y_train == 1).mean()),
        "test_positive_rate": float((y_test == 1).mean()),
        "train_indices_sha256": _hash_index(X_train.index),
        "test_indices_sha256": _hash_index(X_test.index),
    }
    with (splits_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata

# Phan 5: Ham doc cho M04 va khoa tap test cho M06
def _load_split_csv(
    split_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run `python -m src.data_split` first."
        )
    df = pd.read_csv(split_path, index_col=ROW_INDEX_NAME)
    df.index.name = None
    return validate_schema(df, config)


def load_train_data(
    splits_dir: Path | None = None, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the persisted 80% training split (X_train, y_train)."""
    if config is None:
        config = load_config()
    target_dir = Path(splits_dir) if splits_dir is not None else ROOT / config["paths"]["splits"]
    return _load_split_csv(target_dir / "train.csv", config)


def load_test_data(
    splits_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    confirm_final_evaluation: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the sealed 20% test split (M06 final evaluation only)."""
    if not confirm_final_evaluation:
        raise RuntimeError(
            "Test set is sealed until M06 final evaluation. "
            "Pass confirm_final_evaluation=True only when running locked M06 evaluation."
        )
    if config is None:
        config = load_config()
    target_dir = Path(splits_dir) if splits_dir is not None else ROOT / config["paths"]["splits"]
    return _load_split_csv(target_dir / "test.csv", config)


def main() -> int:
    config = load_config()
    clean_path = ROOT / config["paths"]["processed"] / "diabetes_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {clean_path}. Run `python -m src.cleaning` first."
        )
    df = pd.read_csv(clean_path)
    metadata = save_splits(df, config=config, source_path=clean_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

def _load_split_csv(
    split_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run `python -m src.data_split` first."
        )
    df = pd.read_csv(split_path, index_col=ROW_INDEX_NAME)
    df.index.name = None
    return validate_schema(df, config)


def load_train_data(
    splits_dir: Path | None = None, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the persisted 80% training split (X_train, y_train)."""
    if config is None:
        config = load_config()
    target_dir = Path(splits_dir) if splits_dir is not None else ROOT / config["paths"]["splits"]
    return _load_split_csv(target_dir / "train.csv", config)


def load_test_data(
    splits_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    confirm_final_evaluation: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the sealed 20% test split (M06 final evaluation only)."""
    if not confirm_final_evaluation:
        raise RuntimeError(
            "Test set is sealed until M06 final evaluation. "
            "Pass confirm_final_evaluation=True only when running locked M06 evaluation."
        )
    if config is None:
        config = load_config()
    target_dir = Path(splits_dir) if splits_dir is not None else ROOT / config["paths"]["splits"]
    return _load_split_csv(target_dir / "test.csv", config)


def main() -> int:
    config = load_config()
    clean_path = ROOT / config["paths"]["processed"] / "diabetes_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {clean_path}. Run `python -m src.cleaning` first."
        )
    df = pd.read_csv(clean_path)
    metadata = save_splits(df, config=config, source_path=clean_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())