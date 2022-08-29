from abc import ABC, abstractmethod

import numpy as np

from fruits.scope import Seed


class FeatureSieve(Seed, ABC):
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
        self._fit(X, **kwargs)

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._transform(X, **kwargs)

    def _get_cache_keys(self) -> dict[str, list[str]]:
        # returns keys for cache needed in the sieve
        return dict()

    @abstractmethod
    def _summary(self) -> str:
        ...

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        return self._summary()
