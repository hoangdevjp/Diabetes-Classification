"""Check the M01 environment with synthetic data; never read the real dataset."""

from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import matplotlib
import nbformat
import numpy as np
import pandas as pd
import scipy
import seaborn
import sklearn
import streamlit
import yaml
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11.x for the shared project environment.")

    with (ROOT / "configs/experiment.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["split"] != {"test_size": 0.2, "stratify": True, "random_state": 42}:
        raise ValueError("Split convention differs from the approved roadmap.")
    if config["cv"] != {
        "strategy": "StratifiedKFold", "n_splits": 5,
        "shuffle": True, "random_state": 42,
    }:
        raise ValueError("CV convention differs from the approved roadmap.")
    if config["training"]["primary_metric"] != "average_precision":
        raise ValueError("The shared primary metric must be average_precision.")

    # Exercise NumPy/SciPy/sklearn binaries with synthetic records only.
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    pipeline = make_pipeline(StandardScaler(), GaussianNB()).fit(X, y)
    scores = pipeline.predict_proba(X[:3])
    if scores.shape != (3, 2) or not np.isfinite(scores).all():
        raise RuntimeError("Synthetic pipeline smoke check failed.")

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise RuntimeError("No project notebooks found.")
    for path in notebooks:
        nbformat.validate(nbformat.read(path, as_version=4))

    packages = (
        "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib",
        "seaborn", "PyYAML", "jupyterlab", "ipykernel", "nbformat", "streamlit",
    )
    print(f"Python {sys.version.split()[0]}")
    for package in packages:
        print(f"  {package}=={version(package)}")
    print(f"OK: config, synthetic pipeline and {len(notebooks)} notebook schemas.")
    print("Real dataset and final test set were not opened.")


if __name__ == "__main__":
    main()
