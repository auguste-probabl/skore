"""Types between parts of the sklearn module."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol, Union

from numpy.typing import ArrayLike

MLTask = Literal[
    "binary-classification",
    "classification",
    "clustering",
    "multiclass-classification",
    "multioutput-binary-classification",
    "multioutput-multiclass-classification",
    "multioutput-regression",
    "regression",
    "unknown",
]


class _DefaultType:
    """Sentinel class for default values."""

    def __repr__(self) -> str:
        return "<DEFAULT>"


_DEFAULT = _DefaultType()
PositiveLabel = Union[int, float, bool, str, None, _DefaultType]
Aggregate = Union[Literal["mean", "std"], Sequence[Literal["mean", "std"]]]


@dataclass
class YPlotData:
    """Response values, either `y_true` or `y_pred`.

    Used for passing to Display classes.
    """

    estimator_name: str
    split_index: Optional[int]
    y: ArrayLike


ReportType = Literal[
    "cross-validation",
    "estimator",
    "comparison-estimator",
    "comparison-cross-validation",
]


class SKLearnScorer(Protocol):
    """Protocol defining the interface of scikit-learn's _BaseScorer."""

    _score_func: Callable
    _response_method: Union[str, list[str]]
    _kwargs: dict[str, Any]


ScoringName = Union[str, None]

Scoring = Union[str, Callable, SKLearnScorer]


class SKLearnCrossValidator(Protocol):
    """Protocol defining the interface of scikit-learn's cross-validation splitters."""

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""

    def split(
        self, X: ArrayLike, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[ArrayLike, ArrayLike]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
