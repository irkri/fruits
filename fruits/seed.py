from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from .cache import SharedSeedCache

TCopy = TypeVar("TCopy", bound="Seed")


class Seed(ABC):
    """Abstract class for all additional objects that can be added to a
    fruit including preparateurs, words and sieves.
    """

    _cache: SharedSeedCache
    # variable that checks if the seed is used outside a fruit,
    # the _cache will be reset each time a fit is called if no _cache
    # is present at the Seed.fit call
    _cache_at_fit: bool

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    def fit(self, X: np.ndarray) -> None:
        """Fits the seed to the given data."""
        self._cache_at_fit = not hasattr(self, "_cache")
        if self._cache_at_fit:
            self._cache = SharedSeedCache()
        self._fit(X)

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the given data and returns the results."""
        result = self._transform(X)
        if self._cache_at_fit:
            del self._cache
            self._cache_at_fit = False
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Equivalent of calling ``fit`` and ``transform`` consecutively
        on this object.
        """
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def _copy(self: TCopy) -> TCopy:
        ...

    def copy(self: TCopy) -> TCopy:
        """Returns a copy of this seed."""
        return self._copy()

    def __str__(self) -> str:
        return self.__class__.__name__
