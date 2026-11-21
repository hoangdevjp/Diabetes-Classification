"""Unit tests for M03 data split, preprocessing pipeline, leakage guard, and metrics schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data_split import (
    ROOT,
    compute_dataframe_hash,
    load_config,
    load_test_data,
    load_train_data,
    save_splits,
    split_train_test,
    validate_schema,
)
from src.evaluation import (
    REQUIRED_CV_KEYS,
    build_metrics_record,
    compute_classification_metrics,
    evaluate_cv,
    get_continuous_scores,
    save_metrics_json,
    validate_metrics_record,
)
from src.preprocessing import build_pipeline, build_preprocessor, get_cv_splitter


def make_synthetic_diabetes_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame matching the 21 confirmed predictors + target."""
    config = load_config()
    rng = np.random.default_rng(seed)
    features = config["dataset"]["feature_columns"]
    data: dict[str, np.ndarray] = {}
    for col in features:
        if col == "BMI":
            data[col] = rng.integers(15, 50, size=n_rows)
        elif col in {"MentHlth", "PhysHlth"}:
            data[col] = rng.integers(0, 31, size=n_rows)
        elif col in {"GenHlth"}:
            data[col] = rng.integers(1, 6, size=n_rows)
        elif col == "Age":
            data[col] = rng.integers(1, 14, size=n_rows)
        elif col == "Education":
            data[col] = rng.integers(1, 7, size=n_rows)
        elif col == "Income":
            data[col] = rng.integers(1, 9, size=n_rows)
        else:
            data[col] = rng.integers(0, 2, size=n_rows)

    logits = 0.08 * (data["BMI"] - 27) + 0.9 * data["HighBP"] + 0.5 * data["GenHlth"] - 2.4
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_rows) < probs).astype("int64")
    # Guarantee both classes have enough rows for 5-fold stratified CV.
    y[:30] = 1
    y[30:60] = 0
    data[config["dataset"]["target_column"]] = y
    return pd.DataFrame(data)


class TestDataSplitAndSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.df = make_synthetic_diabetes_df(200)

    def test_validate_schema_canonical_order_and_rejections(self) -> None:
        shuffled_cols = list(reversed(self.df.columns.tolist()))
        X, y = validate_schema(self.df[shuffled_cols], self.config)
        self.assertEqual(list(X.columns), self.config["dataset"]["feature_columns"])
        self.assertEqual(set(y.unique()), {0, 1})

        with self.assertRaises(ValueError):
            validate_schema(self.df.drop(columns=["Diabetes_binary"]), self.config)

        bad_nan = self.df.copy()
        bad_nan.loc[0, "Diabetes_binary"] = np.nan
        with self.assertRaises(ValueError):
            validate_schema(bad_nan, self.config)

        bad_label = self.df.copy()
        bad_label.loc[0, "Diabetes_binary"] = 2
        with self.assertRaises(ValueError):
            validate_schema(bad_label, self.config)

        with self.assertRaises(ValueError):
            validate_schema(self.df.drop(columns=["BMI"]), self.config)

        with_id = self.df.copy()
        with_id["ID"] = np.arange(len(with_id))
        with self.assertRaises(ValueError):
            validate_schema(with_id, self.config)

    def test_stratified_split_reproducible_no_overlap_and_guards(self) -> None:
        X_tr1, X_te1, y_tr1, y_te1 = split_train_test(self.df, self.config)
        X_tr2, X_te2, y_tr2, y_te2 = split_train_test(self.df, self.config)

        self.assertEqual(len(X_tr1), 160)
        self.assertEqual(len(X_te1), 40)
        self.assertTrue(X_tr1.index.intersection(X_te1.index).empty)
        self.assertAlmostEqual(float(y_tr1.mean()), float(y_te1.mean()), delta=0.02)
        pd.testing.assert_index_equal(X_tr1.index, X_tr2.index)
        pd.testing.assert_index_equal(X_te1.index, X_te2.index)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            meta = save_splits(self.df, self.config, output_dir=tmp_path)
            self.assertEqual(meta["dataframe_sha256"], compute_dataframe_hash(self.df))
            self.assertEqual(meta["n_train"], 160)
            self.assertEqual(meta["n_test"], 40)

            X_tr_loaded, y_tr_loaded = load_train_data(tmp_path, self.config)
            pd.testing.assert_frame_equal(X_tr1, X_tr_loaded)
            pd.testing.assert_series_equal(y_tr1, y_tr_loaded)

            with self.assertRaises(RuntimeError):
                load_test_data(tmp_path, self.config)

            X_te_loaded, y_te_loaded = load_test_data(
                tmp_path, self.config, confirm_final_evaluation=True
            )
            pd.testing.assert_frame_equal(X_te1, X_te_loaded)
            pd.testing.assert_series_equal(y_te1, y_te_loaded)


class TestPreprocessingAndLeakage(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.df = make_synthetic_diabetes_df(200)
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(
            self.df, self.config
        )

    def test_scaler_selection_by_model_type_and_override(self) -> None:
        def _num_steps(ct: object) -> dict:
            transformers = {name: tr for name, tr, _ in ct.transformers}
            return dict(transformers["num"].steps)

        for scaled_model in ("knn", "svm", "linear_svm", "rbf_svm"):
            ct = build_preprocessor(self.X_train, model_type=scaled_model)
            num_steps = _num_steps(ct)
            self.assertIn("scaler", num_steps)
            self.assertIsInstance(num_steps["scaler"], StandardScaler)

        for unscaled_model in ("random_forest", "naive_bayes", "gaussian_nb"):
            ct = build_preprocessor(self.X_train, model_type=unscaled_model)
            num_steps = _num_steps(ct)
            self.assertNotIn("scaler", num_steps)

        ct_override = build_preprocessor(self.X_train, model_type="random_forest", scale_numeric=True)
        self.assertIn("scaler", _num_steps(ct_override))

        with self.assertRaises(ValueError):
            build_preprocessor(self.X_train, model_type="unknown_model")

    def test_missing_and_categorical_encoding_handle_unknown(self) -> None:
        X_mixed = pd.DataFrame(
            {
                "num_col": [10.0, np.nan, 30.0, 40.0],
                "cat_col": ["a", "b", None, "a"],
            }
        )
        ct = build_preprocessor(X_mixed, model_type="knn")
        transformed_train = ct.fit_transform(X_mixed)
        self.assertFalse(np.isnan(transformed_train).any())

        X_unseen = pd.DataFrame({"num_col": [25.0], "cat_col": ["unseen_category"]})
        transformed_test = ct.transform(X_unseen)
        self.assertEqual(transformed_test.shape[1], transformed_train.shape[1])
        self.assertFalse(np.isnan(transformed_test).any())

    def test_no_test_or_fold_data_leakage(self) -> None:
        pipe_clean = build_pipeline(KNeighborsClassifier(), self.X_train, model_type="knn")
        pipe_clean.fit(self.X_train, self.y_train)

        # Extreme outliers in X_test must never alter fitted training statistics.
        X_test_corrupted = self.X_test.copy()
        X_test_corrupted["BMI"] = 10000
        pipe_with_corrupted_test = build_pipeline(KNeighborsClassifier(), self.X_train, model_type="knn")
        pipe_with_corrupted_test.fit(self.X_train, self.y_train)

        scaler_a = pipe_clean.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
        scaler_b = (
            pipe_with_corrupted_test.named_steps["preprocessor"]
            .named_transformers_["num"]
            .named_steps["scaler"]
        )
        np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_)
        np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_)

        # Inside 5-fold CV, each fold's scaler must fit strictly on its 4 training folds.
        cv = get_cv_splitter(self.config)
        pipe_cv = build_pipeline(KNeighborsClassifier(), self.X_train, model_type="knn")
        cv_out = cross_validate(pipe_cv, self.X_train, self.y_train, cv=cv, return_estimator=True)
        bmi_idx = list(self.X_train.columns).index("BMI")
        full_train_bmi_mean = float(self.X_train["BMI"].mean())

        fold_means_differ_from_full = []
        for (train_idx, _), est in zip(cv.split(self.X_train, self.y_train), cv_out["estimator"]):
            fold_scaler = est.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
            expected_fold_mean = float(self.X_train.iloc[train_idx]["BMI"].mean())
            self.assertAlmostEqual(float(fold_scaler.mean_[bmi_idx]), expected_fold_mean, places=10)
            fold_means_differ_from_full.append(not np.isclose(expected_fold_mean, full_train_bmi_mean))
        self.assertTrue(any(fold_means_differ_from_full))


class TestEvaluationAndMetricsContract(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.df = make_synthetic_diabetes_df(200)
        self.X_train, _, self.y_train, _ = split_train_test(self.df, self.config)

    def test_evaluate_cv_and_save_metrics_json_for_proba_and_decision_models(self) -> None:
        models = [
            ("gaussian_nb", "naive_bayes", GaussianNB()),
            ("knn", "knn", KNeighborsClassifier(n_neighbors=5)),
            ("random_forest", "random_forest", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("linear_svm", "linear_svm", LinearSVC(random_state=42, max_iter=2000)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            for model_name, model_type, estimator in models:
                pipe = build_pipeline(estimator, self.X_train, model_type=model_type)
                cv_summary = evaluate_cv(pipe, self.X_train, self.y_train, config=self.config)
                val_cv = cv_summary["validation_cv"]
                self.assertEqual(set(val_cv.keys()), set(REQUIRED_CV_KEYS))
                self.assertTrue(all(np.isfinite(val_cv[k]) for k in REQUIRED_CV_KEYS))

                pipe.fit(self.X_train, self.y_train)
                scores, score_method = get_continuous_scores(pipe, self.X_train)
                expected_method = "decision_function" if model_name == "linear_svm" else "predict_proba"
                self.assertEqual(score_method, expected_method)
                single_metrics = compute_classification_metrics(
                    self.y_train, pipe.predict(self.X_train), scores
                )
                self.assertIn("average_precision", single_metrics)

                record = build_metrics_record(
                    model_name=model_name,
                    validation_cv=val_cv,
                    best_params={"example": 1},
                    training_time_seconds=cv_summary["training_time_seconds"],
                    baseline_ap_mean=val_cv["average_precision_mean"],
                    score_method=score_method,
                    config=self.config,
                )
                out_file = save_metrics_json(record, tmp_dir / f"{model_name}.json", config=self.config)
                loaded = json.loads(out_file.read_text(encoding="utf-8"))
                validate_metrics_record(loaded, config=self.config)

    def test_metrics_validation_rejects_missing_or_nan_or_discrete_only_estimator(self) -> None:
        pipe = build_pipeline(GaussianNB(), self.X_train, model_type="naive_bayes")
        cv_summary = evaluate_cv(pipe, self.X_train, self.y_train, config=self.config)
        record = build_metrics_record(
            model_name="gaussian_nb",
            validation_cv=cv_summary["validation_cv"],
            best_params={},
            training_time_seconds=cv_summary["training_time_seconds"],
            config=self.config,
        )

        bad_missing = dict(record)
        bad_missing["validation_cv"] = dict(record["validation_cv"])
        del bad_missing["validation_cv"]["average_precision_mean"]
        with self.assertRaises(ValueError):
            validate_metrics_record(bad_missing, config=self.config)

        bad_nan = dict(record)
        bad_nan["validation_cv"] = dict(record["validation_cv"])
        bad_nan["validation_cv"]["roc_auc_mean"] = float("nan")
        with self.assertRaises(ValueError):
            validate_metrics_record(bad_nan, config=self.config)

        class DiscreteOnlyModel:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(X), dtype=int)

        with self.assertRaises(TypeError):
            get_continuous_scores(DiscreteOnlyModel(), self.X_train)


CLEAN_DATA_PATH = ROOT / "data" / "processed" / "diabetes_clean.csv"


class TestCleanedDatasetIntegration(unittest.TestCase):
    @unittest.skipUnless(
        CLEAN_DATA_PATH.exists(),
        "Real cleaned dataset not present locally (expected on clean CI checkout).",
    )
    def test_real_cleaned_dataset_schema_and_split_counts(self) -> None:
        config = load_config()
        df = pd.read_csv(CLEAN_DATA_PATH)
        X, y = validate_schema(df, config)
        self.assertEqual(X.shape, (229474, 21))
        self.assertEqual(len(y), 229474)

        X_train, X_test, y_train, y_test = split_train_test(df, config)
        self.assertEqual(len(X_train), 183579)
        self.assertEqual(len(X_test), 45895)
        self.assertTrue(X_train.index.intersection(X_test.index).empty)
        self.assertAlmostEqual(float(y_train.mean()), float(y_test.mean()), delta=1e-4)


if __name__ == "__main__":
    unittest.main()
