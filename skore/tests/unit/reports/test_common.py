import jedi
import numpy as np
import pytest


def custom_metric(y_true, y_pred, threshold=0.5):
    residuals = y_true - y_pred
    return np.mean(np.where(residuals < threshold, residuals, 1))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_regression",
        "cross_validation_reports_regression",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
    ],
)
class TestCustomMetricSummarize:
    def test_works(self, fixture_name, request):
        """Check that computing a custom metric via register + summarize works."""
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        report.metrics.custom_metric(
            metric_function=custom_metric,
            response_method="predict",
        )

    def test_works_with_kwargs(self, fixture_name, request):
        """Check that computing a custom metric with extra kwargs works."""
        report = request.getfixturevalue(fixture_name)
        if isinstance(report, tuple):
            report = report[0]

        report.metrics.custom_metric(
            metric_function=custom_metric,
            response_method="predict",
            threshold=0.3,
        )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "estimator_reports_regression",
        "cross_validation_reports_regression",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
    ],
)
def test_ipython_completion(fixture_name, request):
    """Non-regression test for #2386.

    We get no tab completions from IPython if jedi raises an exception, so we
    check here that jedi can produce completions without errors.
    """
    report = request.getfixturevalue(fixture_name)
    if isinstance(report, tuple):
        report = report[0]
    interp = jedi.Interpreter("r.", [{"r": report}])
    interp.complete(line=1, column=2)
