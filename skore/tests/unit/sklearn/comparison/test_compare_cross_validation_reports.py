import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationComparisonReport, CrossValidationReport


def test_metrics_binary_classification():
    """Check the metrics work."""
    X, y = make_classification(random_state=42)
    cv_report = CrossValidationReport(LogisticRegression(), X, y)

    comp = CrossValidationComparisonReport([cv_report, cv_report])

    result = comp.metrics.accuracy()
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)
