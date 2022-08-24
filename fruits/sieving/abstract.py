from abc import ABC, abstractmethod

import numpy as np


class FeatureSieve(ABC):
    """Abstract class for a feature sieve. Sieves are the last part of a
    :class:`~fruits.core.fruit.Fruit`.

    A feature sieve is used to transforms a two-dimensional numpy
    array containing iterated sums into a one dimensional numpy array of
    features.

    Args:
        name (str, optional): Identification string of the feature
            sieve. Defaults to an empty string.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def nfeatures(self) -> int:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the sieve to the dataset.

        Args:
            X (np.ndarray): 2-dimensional array of iterated sums.
        """

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transforms the given timeseries dataset."""

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Equivalent of calling ``FeatureSieve.fit`` and
        ``FeatureSieve.transform`` consecutively.
        """
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    def _get_cache_keys(self) -> dict[str, list[str]]:
        # returns keys for cache needed in the sieve
        return dict()

    @abstractmethod
    def copy(self) -> "FeatureSieve":
        pass

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        return "FeatureSieve('" + self.name + "')"

    def __copy__(self) -> "FeatureSieve":
        return self.copy()

    def __repr__(self) -> str:
        return f"fruits.sieving.abstract.FeatureSieve('{self.name}')"
