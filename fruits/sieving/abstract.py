from abc import ABC, abstractmethod

import numpy as np


class FeatureSieve(ABC):
    """Abstract class for a feature sieve. Sieves are the last part of a
    :class:`~fruits.core.fruit.Fruit`.

    A feature sieve is used to transforms a two-dimensional numpy
    array containing iterated sums into a one dimensional numpy array of
    features.
    """

    @abstractmethod
    def _nfeatures(self) -> int:
        ...

    def nfeatures(self) -> int:
        return self._nfeatures()

    def _fit(self, X: np.ndarray, **kwargs) -> None:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the sieve to the dataset.

        Args:
            X (np.ndarray): 2-dimensional array of iterated sums.
        """
        self._fit(X, **kwargs)

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transforms the given timeseries dataset."""
        return self._transform(X, **kwargs)

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
    def _copy(self) -> "FeatureSieve":
        ...

    def copy(self) -> "FeatureSieve":
        """Returns a copy of this feature sieve."""
        return self._copy()

    @abstractmethod
    def _summary(self) -> str:
        ...

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        return self._summary()

    def __copy__(self) -> "FeatureSieve":
        return self.copy()
