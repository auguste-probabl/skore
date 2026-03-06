"""Tests for metrics registry functionality."""

import pickle

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    get_scorer,
    make_scorer,
    mean_squared_error,
    precision_score,
)

from skore import EstimatorReport
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def business_loss(y_true, y_pred, cost_fp, cost_fn):
    """Custom business metric: weighted cost of false positives and negatives."""
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn


custom_scorer = make_scorer(
    business_loss,
    greater_is_better=False,
    response_method="predict",
    cost_fp=10,
    cost_fn=5,
)


def detection_failure_cost(y_true, y_pred_proba, threshold=0.5):
    """Custom metric based on probability threshold."""
    y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
    return business_loss(y_true, y_pred, cost_fp=10, cost_fn=5)


@pytest.fixture
def binary_classification_report(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label=1,
    )


@pytest.fixture
def regression_report(linear_regression_with_train_test):
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


class TestBasicRegistration:
    """Test basic metric registration functionality."""

    def test_register_simple_scorer(self, binary_classification_report):
        """Test registering a simple custom scorer."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        metric = report._metric_registry["business_loss"]

        assert metric.name == "business_loss"
        assert metric.verbose_name == "Business Loss"
        assert metric.greater_is_better is False
        assert metric.response_method == "predict"

    def test_register_multiple_metrics(self, binary_classification_report):
        """Test registering multiple custom metrics."""
        report = binary_classification_report

        scorer1 = custom_scorer
        scorer2 = make_scorer(
            accuracy_score,
            response_method="predict",
        )

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        assert "business_loss" in report._metric_registry
        assert "accuracy_score" in report._metric_registry


class TestSummarizeIntegration:
    """Test integration with the summarize() method."""

    def test_summarize_includes_registered_metrics(self, binary_classification_report):
        """Test that summarize() automatically includes registered metrics."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        display = report.metrics.summarize()

        # Should include both built-in and custom metrics
        assert "Accuracy" in display.frame().index
        assert "Business Loss" in display.frame().index

    def test_summarize_with_explicit_custom_metric(self, binary_classification_report):
        """Test calling summarize with explicit custom metric name."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        # Should be able to call by name
        display = report.metrics.summarize(metric="business_loss")

        assert len(display.data) == 1
        row = display.data.iloc[0]
        assert row["metric"] == "Business Loss"
        assert row["favorability"] == "(↘︎)"

    def test_summarize_with_mixed_metrics(self, binary_classification_report):
        """Test summarize with both built-in and custom metrics."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        # Should work with list including both types
        display = report.metrics.summarize(metric=["accuracy", "business_loss"])

        assert set(display.data["metric"]) == {"Accuracy", "Business Loss"}


class TestBuiltInProtection:
    """Test protection against overriding built-in metric names."""

    def test_cannot_override_builtin_metric(self, binary_classification_report):
        """Test that registering with a built-in technical name raises an error."""
        report = binary_classification_report

        # Try to register a scorer with a built-in metric name
        fake_accuracy = make_scorer(
            lambda y_true, y_pred: 1.0,
            response_method="predict",
        )
        # Force the name to conflict with built-in technical name
        fake_accuracy._score_func.__name__ = "accuracy"

        err_msg = "Cannot register 'accuracy': it is a built-in metric name."
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.register(fake_accuracy)


class TestScorerExtraction:
    """Test extraction of metadata from sklearn scorers."""

    def test_extract_score_func(self, binary_classification_report):
        """Test that _score_func is correctly extracted."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        # The registered metric should use the original function
        metric = report._metric_registry["business_loss"]
        assert metric.score_func == business_loss

    def test_extract_response_method(self, binary_classification_report):
        """Test that _response_method is correctly extracted."""
        report = binary_classification_report

        # Scorer using predict_proba
        proba_scorer = make_scorer(
            detection_failure_cost,
            greater_is_better=False,
            response_method="predict_proba",
            threshold=0.7,
        )
        report.metrics.register(proba_scorer)

        metric = report._metric_registry["detection_failure_cost"]
        assert metric.response_method == "predict_proba"

    def test_extract_kwargs(self, binary_classification_report):
        """Test that scorer kwargs are correctly extracted and forwarded."""
        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
            cost_fp=20,
            cost_fn=3,
        )
        report.metrics.register(scorer)

        display = report.metrics.summarize(metric="business_loss")
        assert display.data["score"].notna().all()


class TestCacheBehavior:
    """Test caching behavior with registered metrics."""

    def test_metric_result_is_cached(self, binary_classification_report):
        """Test that metric results are cached after first computation."""
        report = binary_classification_report

        def counting_metric(y_true, y_pred):
            return accuracy_score(y_true, y_pred)

        scorer = make_scorer(counting_metric, response_method="predict")
        report.metrics.register(scorer)

        with check_cache_changed(report._cache):
            report.metrics.summarize(metric="counting_metric")

        with check_cache_unchanged(report._cache):
            report.metrics.summarize(metric="counting_metric")

    def test_reregister_invalidates_cache(self, binary_classification_report):
        """Test that re-registering a metric invalidates its cache only."""
        report = binary_classification_report

        def metric1(y_true, y_pred):
            return 0.1

        def metric2(y_true, y_pred):
            return 0.2

        scorer1 = make_scorer(metric1, response_method="predict")
        scorer2 = make_scorer(metric2, response_method="predict")

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        # Compute both - should cache
        report.metrics.summarize(metric="metric1")
        report.metrics.summarize(metric="metric2")

        # Re-register metric1 with a new function
        def metric1(y_true, y_pred):
            return 0.3

        scorer1_v2 = make_scorer(metric1, response_method="predict")
        report.metrics.register(scorer1_v2)

        # metric2 should use cache
        with check_cache_unchanged(report._cache):
            result2 = report.metrics.summarize(metric="metric2")

        # metric1 should compute fresh
        with check_cache_changed(report._cache):
            result1 = report.metrics.summarize(metric="metric1")

        assert result1.data["score"].iloc[0] == 0.3
        assert result2.data["score"].iloc[0] == 0.2

    def test_different_metrics_have_separate_cache(self, binary_classification_report):
        """Test that different metrics don't share cache entries."""
        report = binary_classification_report

        def metric1(y_true, y_pred):
            return 0.1

        def metric2(y_true, y_pred):
            return 0.9

        scorer1 = make_scorer(metric1, response_method="predict")
        scorer2 = make_scorer(metric2, response_method="predict")

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        result1 = report.metrics.summarize(metric="metric1")
        result2 = report.metrics.summarize(metric="metric2")

        assert result1.data["score"].iloc[0] == 0.1
        assert result2.data["score"].iloc[0] == 0.9


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_on_report_without_train_data(
        self, logistic_binary_classification_with_train_test
    ):
        """Test that registration works even without train data."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(accuracy_score, response_method="predict")
        report.metrics.register(scorer)

        with pytest.raises(ValueError, match="(?i)train|data"):
            report.metrics.summarize(metric="accuracy_score", data_source="train")

    def test_register_scorer_with_incompatible_response_method(
        self, svc_binary_classification_with_train_test
    ):
        """Test error when scorer's response_method is incompatible with estimator."""
        estimator, X_train, X_test, y_train, y_test = (
            svc_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        incompatible_scorer = make_scorer(
            accuracy_score,
            response_method="predict_proba",
        )

        report.metrics.register(incompatible_scorer)

        err_msg = "SVC has none of the following attributes: predict_proba."
        with pytest.raises(AttributeError, match=err_msg):
            report.metrics.summarize(metric="accuracy_score")

    def test_register_duplicate_name_replaces(self, binary_classification_report):
        """Test that registering with duplicate name silently replaces."""
        report = binary_classification_report

        def score(y_true, y_pred):
            return 0

        report.metrics.register(make_scorer(score, response_method="predict"))

        nb_metrics_before_overwriting = len(report._metric_registry)

        result = report.metrics.summarize(metric="score")
        assert result.data["score"].iloc[0] == 0

        # Register a new metric with the same name
        def score(y_true, y_pred):
            return 1

        report.metrics.register(make_scorer(score, response_method="predict"))

        assert len(report._metric_registry) == nb_metrics_before_overwriting

        # `summarize` contains only the new output
        result = report.metrics.summarize(metric="score")
        assert result.data["score"].iloc[0] == 1


class TestDifferentMLTasks:
    """Test that registry works across different ML tasks."""

    def test_register_multiclass(
        self, logistic_multiclass_classification_with_train_test
    ):
        """Test registration of multiclass-compatible metric."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_multiclass_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(accuracy_score, response_method="predict")
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Accuracy Score" in display.data["metric"].values

    def test_register_regression(self, regression_report):
        """Test registration on regression report."""
        report = regression_report

        def custom_mse(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)

        scorer = make_scorer(
            custom_mse,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Custom Mse" in display.data["metric"].values

    def test_register_multioutput_regression(
        self, linear_regression_multioutput_with_train_test
    ):
        """Test registration on multioutput regression."""
        estimator, X_train, X_test, y_train, y_test = (
            linear_regression_multioutput_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Mean Squared Error" in display.data["metric"].values

    def test_register_wrong_ml_task(self, linear_regression_with_train_test):
        """Registering a metric incompatible with the ML task doesn't crash."""
        estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
        report = EstimatorReport(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        scorer = make_scorer(
            mean_squared_error, greater_is_better=False, response_method="predict"
        )
        report.metrics.register(scorer)


class TestDictReturnValues:
    """Test that metrics returning dicts work correctly (per-label scores).

    Note: Multimetric scorers (single scorer returning multiple different metrics)
    are NOT supported - users should register metrics separately.
    """

    def test_per_class_accuracy_dict(self, binary_classification_report):
        """Test metric that returns per-class scores as dict."""
        report = binary_classification_report

        def per_class_accuracy(y_true, y_pred):
            """Return accuracy for each class."""
            accuracies = {}
            for label in np.unique(y_true):
                mask = y_true == label
                accuracies[int(label)] = float((y_pred[mask] == label).mean())
            return accuracies

        scorer = make_scorer(per_class_accuracy, response_method="predict")
        report.metrics.register(scorer)

        display = report.metrics.summarize(metric="per_class_accuracy")

        metric_rows = display.data[display.data["metric"] == "Per Class Accuracy"]
        assert len(metric_rows) == 2
        assert set(metric_rows["label"].values) == {0, 1}

        # Cached correctly
        with check_cache_unchanged(report._cache):
            report.metrics.summarize(metric="per_class_accuracy")

    def test_multimetric_scorer_not_recommended(self, binary_classification_report):
        """Multimetric scorers are treated as per-label scores (not supported)."""
        report = binary_classification_report

        def multimetric_scorer(y_true, y_pred):
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary"),
            }

        report.metrics.register(
            make_scorer(multimetric_scorer, response_method="predict")
        )

        display = report.metrics.summarize(metric="multimetric_scorer")

        metric_rows = display.data[display.data["metric"] == "Multimetric Scorer"]
        # Not quite right...
        assert set(metric_rows["label"]) == {"accuracy", "precision"}


class TestStringScorerNames:
    """Test support for string scorer names (via sklearn.metrics.get_scorer)."""

    def test_register_with_string_scorer_name(self, binary_classification_report):
        """Test registering a metric using its sklearn string name."""
        report = binary_classification_report

        report.metrics.register("f1")
        assert "f1_score" in report._metric_registry

        # NOTE: Not "f1" (that would re-run sklearn.metrics.get_scorer())
        display = report.metrics.summarize(metric="f1_score")
        metric_rows = display.data[display.data["metric"] == "F1 Score"]

        assert len(metric_rows) == 1

    def test_string_scorer_appears_in_summarize(self, binary_classification_report):
        """Test that string scorers appear in summarize() output."""
        report = binary_classification_report

        display = report.metrics.summarize()
        metrics_before = set(display.data["metric"])

        report.metrics.register("f1")

        display = report.metrics.summarize()
        metrics_after = set(display.data["metric"])

        assert metrics_after - metrics_before == {"F1 Score"}

    def test_neg_scorer(self, regression_report):
        """Test that neg_* scorers have correct sign, favorability, and display name."""
        report = regression_report

        report.metrics.register(get_scorer("neg_mean_squared_error"))

        # `neg_` was stripped off metric name
        assert "mean_squared_error" in report._metric_registry

        display = report.metrics.summarize(metric="mean_squared_error")
        row = display.data.iloc[0]

        assert row["score"] >= 0
        assert row["favorability"] == "(↘︎)"
        assert not row["metric"].lower().startswith("neg")

    def test_invalid_string_scorer_name(self, binary_classification_report):
        """Test that invalid sklearn scorer names raise an error."""
        report = binary_classification_report

        with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
            report.metrics.register("xyz")


class TestSerialization:
    """Test that registered metrics survive pickling (for Project storage)."""

    def test_pickle_report_with_registered_metric(self, binary_classification_report):
        """Test that registered metrics survive pickle/unpickle with metadata."""
        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
            cost_fp=20,
            cost_fn=3,
        )
        report.metrics.register(scorer)

        report2 = pickle.loads(pickle.dumps(report))

        assert "business_loss" in report2._metric_registry

        metric = report2._metric_registry["business_loss"]
        assert metric.is_callable()
        assert metric.name == "business_loss"
        assert metric.verbose_name == "Business Loss"
        assert metric.greater_is_better is False
        assert metric.response_method == "predict"
        assert metric.kwargs == {"cost_fp": 20, "cost_fn": 3}

        display = report2.metrics.summarize()
        assert "Business Loss" in display.data["metric"].values

    def test_pickle_lambda_warns_and_loses_func(self, binary_classification_report):
        """Lambdas warn at registration and lose their function after pickling."""
        report = binary_classification_report

        scorer = make_scorer(
            lambda y_true, y_pred: 0.5,
            response_method="predict",
        )

        with pytest.warns(UserWarning, match="lambda"):
            report.metrics.register(scorer)

        with pytest.warns(UserWarning, match="could not be restored"):
            report2 = pickle.loads(pickle.dumps(report))

        assert "<lambda>" in report2._metric_registry
        assert not report2._metric_registry["<lambda>"].is_callable()

    def test_source_code_captured_and_survives_pickle(
        self, binary_classification_report
    ):
        """Source code is captured at registration and preserved through pickle."""
        report = binary_classification_report

        report.metrics.register(custom_scorer)

        metric = report._metric_registry["business_loss"]
        assert metric.source_code is not None
        assert "def business_loss" in metric.source_code
        assert "cost_fp" in metric.source_code

        report2 = pickle.loads(pickle.dumps(report))
        metric2 = report2._metric_registry["business_loss"]
        assert metric2.source_code is not None
        assert "def business_loss" in metric2.source_code

    def test_closure_warning(self, binary_classification_report):
        """Closures warn about pickling at registration time."""
        report = binary_classification_report

        multiplier = 2.0

        def closure_metric(y_true, y_pred):
            return np.mean(y_true == y_pred) * multiplier

        scorer = make_scorer(closure_metric, response_method="predict")

        with pytest.warns(UserWarning, match="closure"):
            report.metrics.register(scorer)

    def test_multiple_metrics_pickle(self, binary_classification_report):
        """Multiple registered metrics all survive pickling."""
        report = binary_classification_report

        scorer1 = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
            cost_fp=10,
            cost_fn=5,
        )
        scorer2 = make_scorer(accuracy_score, response_method="predict")

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        report2 = pickle.loads(pickle.dumps(report))

        assert report2._metric_registry["business_loss"].is_callable()
        assert report2._metric_registry["accuracy_score"].is_callable()
