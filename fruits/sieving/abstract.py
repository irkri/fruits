from abc import ABC, abstractmethod

import numpy as np

from fruits.node import FruitNode

class FeatureSieve(FruitNode):
    """Abstract class for a feature sieve. Sieves are the last part of a
    FRUITS pipeline.

    A feature sieve is used to transforms a two-dimensional numpy
    array containing iterated sums into a onedimensional numpy array of
    features. The length of the resulting array can be determined by
    calling ``FeatureSieve.nfeatures``.

    Each class that inherits FeatureSieve must override the methods
    ``FeatureSieve.sieve`` and ``FeatureSieve.nfeatures``.
    """
    def __init__(self, name: str = ""):
        super().__init__(name)

    @abstractmethod
    def nfeatures(self) -> int:
        pass

    def fit(self, X: np.ndarray):
        """Fits the sieve to the dataset.

        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        """
        pass

    @abstractmethod
    def sieve(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_sieve(self, X: np.ndarray) -> np.ndarray:
        """Equivalent of calling ``FeatureSieve.fit`` and
        ``FeatureSieve.sieve`` consecutively.

        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        :returns: Array of features
        :rtype: np.ndarray
        """
        self.fit(X)
        return self.sieve(X)

    @abstractmethod
    def copy(self) -> "FeatureSieve":
        pass

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        return "FeatureSieve('" + self._name + "')"

    def __copy__(self) -> "FeatureSieve":
        return self.copy()

    def __repr__(self) -> str:
        return f"fruits.sieving.abstract.FeatureSieve('{self.name}')"
