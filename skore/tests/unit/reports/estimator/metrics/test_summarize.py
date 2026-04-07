"""Tests for EstimatorReport.metrics.summarize().

Organised by metric input type, then corner cases:

- Default metrics — by ML task variant
- Metric strings — skore built-ins
- Registered metrics — scorers, callables, and mixed via register()
- pos_label
- Cache and data_source
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, MetricsSummaryDisplay
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def check_display_structure(
    display,
    *,
    expected_metrics,
    expected_estimator_name=None,
    expected_data_source="test",
    expected_favorability=None,
    expected_average=None,
):
    """Check the full structure of a MetricsSummaryDisplay.data DataFrame."""
    assert isinstance(display.data, pd.DataFrame)
    data = display.data

    assert set(data.columns) == {
        "metric",
        "estimator_name",
        "data_source",
        "label",
        "average",
        "output",
        "score",
        "favorability",
    }
    assert pd.api.types.is_numeric_dtype(data["score"])
    assert set(data["metric"]) == expected_metrics
    assert set(data["estimator_name"]) == {expected_estimator_name}
    assert set(data["data_source"]) == {expected_data_source}
    if expected_average is None:
        assert data["average"].isna().all()
    else:
        assert set(data["average"]) == expected_average
    if expected_favorability is None:
        expected_favorability = {"(↗︎)", "(↘︎)"}
    assert set(data["favorability"]) == expected_favorability


# Default metrics


def test_default_plain(forest_binary_classification_with_test):
    """metric=None selects the canonical defaults for the ML task."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=None)

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )


def test_default_binary_classification_svc(svc_binary_classification_with_test):
    """SVC (no predict_proba): no ROC AUC, Log loss, or Brier score."""
    estimator, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    assert isinstance(display.data, pd.DataFrame)
    # No Brier score
    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="SVC",
    )


def test_default_multiclass_classification_forest(
    forest_multiclass_classification_with_test,
):
    """Multiclass classification with RandomForestClassifier."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Log loss",
            "Precision",
            "Recall",
            "ROC AUC",
            "Predict time (s)",
            "Fit time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_default_multiclass_classification_svc(svc_multiclass_classification_with_test):
    """Multiclass classification with SVC (no predict_proba)."""
    estimator, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="SVC",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_default_regression(linear_regression_with_test):
    """Regression with LinearRegression."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_default_multioutput_regression(linear_regression_multioutput_with_test):
    """Multioutput regression: default summarize returns aggregated scores."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()


def test_default_without_predict_proba(custom_classifier_no_predict_proba_with_test):
    """Default metrics skip roc_auc, log_loss, and brier_score without predict_proba."""
    estimator, X_test, y_test = custom_classifier_no_predict_proba_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="CustomClassifierPredictOnly",
    )


def test_unknown_ml_task(forest_binary_classification_with_test):
    """Unknown ML task with a registered custom metric."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report._ml_task = "unknown-task"

    def custom_metric(y_true, y_pred):
        return 0.8

    report.metrics.register(custom_metric, response_method="predict")
    display = report.metrics.summarize(metric=["custom_metric"])

    assert len(display.data) == 1
    assert display.data["score"].values[0] == 0.8
    assert display.data["label"].isna().all()
    assert display.data["average"].isna().all()
    assert display.data["output"].isna().all()


def test_invalid_metric_type(linear_regression_with_test):
    """A non-string in the metric list raises a clear error."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with pytest.raises((TypeError, KeyError)):
        report.metrics.summarize(metric=[1])


# Metric string


def test_string_plain(linear_regression_with_test):
    """A list of skore built-in metric strings resolves to correct display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["r2", "rmse"])

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE"},
        expected_estimator_name="LinearRegression",
    )


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_string_unknown(linear_regression_with_test, metric):
    """An unrecognised metric string raises a clear KeyError."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with pytest.raises(KeyError, match="Unknown metric"):
        report.metrics.summarize(metric=[metric])


# Registered metrics — scorers


def test_register_scorer_plain(linear_regression_with_test):
    """Registering sklearn scorers, then summarizing by name."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")
    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    report.metrics.register(r2_scorer)
    report.metrics.register(mae_scorer)

    display = report.metrics.summarize(metric=["r2_score", "mean_absolute_error"])

    assert isinstance(display.data, pd.DataFrame)
    scores = display.data.set_index("metric")["score"]
    assert len(scores) == 2
    assert scores.iloc[0] == pytest.approx(r2_score(y_test, estimator.predict(X_test)))
    assert scores.iloc[1] == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_register_scorer_with_extra_args(forest_binary_classification_with_test):
    """Scorers created with make_scorer embed their own extra kwargs (e.g. average)."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    f1_scorer = make_scorer(f1_score, response_method="predict", average="macro")
    precision_scorer = make_scorer(
        precision_score,
        response_method="predict",
        average="weighted",
        zero_division=0,
    )

    report.metrics.register(f1_scorer)
    report.metrics.register(precision_scorer)

    display = report.metrics.summarize(metric=["f1_score", "precision_score"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2
    averages = display.data.set_index("metric")["average"]
    assert averages.iloc[0] == "macro"
    assert averages.iloc[1] == "weighted"


def test_register_scorer_with_name(linear_regression_with_test):
    """Registering scorers with custom names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")
    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    report.metrics.register(r2_scorer, name="Custom R2")
    report.metrics.register(mae_scorer, name="Custom MAE")

    display = report.metrics.summarize(metric=["Custom R2", "Custom MAE"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2


def test_register_scorer_with_average(forest_multiclass_classification_with_test):
    """Multiclass classification with average parameter via registered scorer."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    scorer = make_scorer(f1_score, average="macro")
    report.metrics.register(scorer)
    display = report.metrics.summarize(metric=["f1_score"])

    assert len(display.data) == 1
    assert display.data["average"].values[0] == "macro"
    assert display.data["label"].isna().all()


def test_register_scorer_response_method(linear_regression_with_test):
    """response_method embedded in the scorer is used automatically.

    Regression test for #2203.
    """
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def business_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    scorer = make_scorer(
        business_loss, greater_is_better=False, response_method="predict"
    )

    report.metrics.register(scorer)
    display = report.metrics.summarize(metric=["business_loss"])

    assert len(display.data) == 1
    expected = business_loss(y_test, estimator.predict(X_test))
    assert display.data["score"].iloc[0] == pytest.approx(expected)


def test_register_scorer_pos_label(forest_binary_classification_with_test):
    """pos_label specified in scorer is used when registering and summarizing."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=0)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    report.metrics.register(f1_scorer)
    display = report.metrics.summarize(metric=["f1_score"])
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1


def test_register_scorer_binary_classification(
    forest_binary_classification_with_test,
):
    """Scorers with pos_label in binary classification via register."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    scorer = make_scorer(f1_score, response_method="predict", average="macro")

    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)

    report.metrics.register(accuracy_score, response_method="predict")
    report.metrics.register(scorer)

    display = report.metrics.summarize(
        metric=["accuracy", "accuracy_score", "f1_score"]
    )
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 3

    expected_scores = [
        accuracy_score(y_test, estimator.predict(X_test)),
        accuracy_score(y_test, estimator.predict(X_test)),
        f1_score(y_test, estimator.predict(X_test), average="macro"),
    ]
    np.testing.assert_allclose(display.data["score"].values, expected_scores)


# Registered metrics — callables


def test_register_callable_plain(linear_regression_with_test):
    """Registering a plain callable, then summarizing by name."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    report.metrics.register(my_mae, response_method="predict")
    display = report.metrics.summarize(metric=["my_mae"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_register_callable_with_extra_args(linear_regression_with_test):
    """Callables with extra kwargs registered via register(**kwargs)."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    weights = np.ones_like(y_test) * 3.0

    def weighted_mae(y_true, y_pred, weights):
        return float(np.average(np.abs(y_true - y_pred), weights=weights))

    report.metrics.register(weighted_mae, response_method="predict", weights=weights)
    display = report.metrics.summarize(metric=["weighted_mae"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        weighted_mae(y_test, estimator.predict(X_test), weights)
    )


def test_register_callable_with_name(linear_regression_with_test):
    """Callable registered with a custom name."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def my_metric(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    report.metrics.register(
        my_metric, name="My Custom Error", response_method="predict"
    )
    display = report.metrics.summarize(metric=["My Custom Error"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    score = display.data["score"].values[0]
    assert score == pytest.approx(
        mean_absolute_error(y_test, estimator.predict(X_test))
    )


def test_register_callable_no_response_method(linear_regression_with_test):
    """Callable registered with default response_method='predict' works."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return 0.75

    report.metrics.register(custom_metric, response_method="predict")
    display = report.metrics.summarize(metric=["custom_metric"])

    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 1
    assert display.data["score"].values[0] == 0.75


# Registered metrics — mixed (register then summarize)


def test_register_mixed(linear_regression_with_test):
    """Register a scorer and callable, then summarize all together with built-ins."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    def my_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    report.metrics.register(mae_scorer)
    report.metrics.register(my_r2, response_method="predict")

    display = report.metrics.summarize(metric=["rmse", "mean_absolute_error", "my_r2"])

    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {"RMSE", "Mean Absolute Error", "My R2"}


def test_register_mixed_with_extra_args(linear_regression_with_test):
    """Register multiple metric types with kwargs, then summarize."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    mae_scorer = make_scorer(
        mean_absolute_error, response_method="predict", greater_is_better=False
    )

    def scaled_mae(y_true, y_pred, scale=1.0):
        return scale * mean_absolute_error(y_true, y_pred)

    report.metrics.register(mae_scorer)
    report.metrics.register(scaled_mae, response_method="predict", scale=2.0)

    display = report.metrics.summarize(
        metric=["r2", "mean_absolute_error", "scaled_mae"]
    )

    assert isinstance(display.data, pd.DataFrame)
    scores = display.data.set_index("metric")["score"]
    plain_mae = mean_absolute_error(y_test, estimator.predict(X_test))
    assert scores["Scaled Mae"] == pytest.approx(2.0 * plain_mae)


def test_register_mixed_with_names(linear_regression_with_test):
    """Register metrics with custom names, then summarize by those names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    r2_scorer = make_scorer(r2_score, response_method="predict")

    def my_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    report.metrics.register(r2_scorer, name="Scorer R2")
    report.metrics.register(my_mae, name="Callable MAE", response_method="predict")

    display = report.metrics.summarize(metric=["rmse", "Scorer R2", "Callable MAE"])

    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {"RMSE", "Scorer R2", "Callable Mae"}


# Summarize with single metric equivalence


def test_single_string_equivalence(forest_binary_classification_with_test):
    """Passing a single metric string is equivalent to passing a list."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display_single = report.metrics.summarize(metric="accuracy")
    display_list = report.metrics.summarize(metric=["accuracy"])
    pd.testing.assert_frame_equal(display_single.data, display_list.data)


# pos_label


def test_pos_label(forest_binary_classification_with_test):
    """pos_label collapses per-class metrics to a single row."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert len(display.data[display.data["metric"] == "Precision"]) == 1
    assert len(display.data[display.data["metric"] == "Recall"]) == 1
    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_pos_label_strings(forest_binary_classification_with_test):
    """Binary classification with string labels."""
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array(["neg", "pos"], dtype=object)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    assert set(labels) == {"neg", "pos"}


def test_pos_label_bool(forest_binary_classification_with_test):
    """Binary classification with boolean labels."""
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array([False, True], dtype=bool)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    assert any(label is np.False_ for label in labels)
    assert any(label is np.True_ for label in labels)


@pytest.mark.parametrize(
    "metric, metric_fn", [("precision", precision_score), ("recall", recall_score)]
)
def test_pos_label_overwrite(metric, metric_fn):
    """pos_label can be set when creating the report."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)

    # Without pos_label - should have multiple rows (one per class)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = report.metrics.summarize(metric=metric)
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2
    assert set(display.data["label"]) == {"A", "B"}

    # With pos_label="B" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_B = display.data["score"].values[0]
    assert score_B == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="B"))

    # With pos_label="A" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_A = display.data["score"].values[0]
    assert score_A == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="A"))


# Cache and data_source


def test_cache(forest_binary_classification_with_test):
    """summarize() results are cached; second call returns the same data."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with check_cache_changed(report._cache):
        result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)

    with check_cache_unchanged(report._cache):
        result_from_cache = report.metrics.summarize()
    assert_frame_equal(result.data, result_from_cache.data)


def test_data_source_both(forest_binary_classification_data):
    """data_source='both' concatenates train and test results."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display_train = report.metrics.summarize(data_source="train")
    display_test = report.metrics.summarize(data_source="test")
    display_both = report.metrics.summarize(data_source="both")

    assert set(display_both.data["data_source"]) == {"train", "test"}

    train_data = display_both.data[display_both.data["data_source"] == "train"]
    assert_array_equal(train_data["score"], display_train.data["score"])

    test_data = display_both.data[display_both.data["data_source"] == "test"]
    assert_array_equal(test_data["score"], display_test.data["score"])
