from abc import abstractmethod
from typing import Dict, List

import numpy as np


class FeatureSieve:
    """Abstract class for a feature sieve. Sieves are the last part of a
    :class:`~fruits.core.fruit.Fruit`.

    A feature sieve is used to transforms a two-dimensional numpy
    array containing iterated sums into a one dimensional numpy array of
    features.

    :param name: Identification string of the feature sieve.,
        defaults to ""
    :type name: str, optional
    """

    name: str

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def nfeatures(self) -> int:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the sieve to the dataset.

        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        """

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Equivalent of calling ``FeatureSieve.fit`` and
        ``FeatureSieve.transform`` consecutively.

        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        :rtype: np.ndarray
        """
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    def _get_cache_keys(self) -> Dict[str, List[str]]:
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
