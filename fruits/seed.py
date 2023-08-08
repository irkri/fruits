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

    @property
    def requires_fitting(self) -> bool:
        return True

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    def fit(self, X: np.ndarray) -> None:
        """Fits the seed to the given data."""
        has_cache = hasattr(self, "_cache")
        if not has_cache:
            self._cache = SharedSeedCache(
                X[:, np.newaxis, :] if X.ndim == 2 else X
            )
        self._fit(X)
        if not has_cache:
            del self._cache

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the given data and returns the results."""
        has_cache = hasattr(self, "_cache")
        if not has_cache:
            self._cache = SharedSeedCache(
                X[:, np.newaxis, :] if X.ndim == 2 else X
            )
        result = self._transform(X)
        if not has_cache:
            del self._cache
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

    def _label(self, index: int = 0) -> str:
        return str(self)

    def label(self, index: int = 0) -> str:
        """Returns the label of a single transform produced by this
        seed.

        Args:
            index (int, optional): The index of one transform if
                multiple are produced. Defaults to the first transform.
        """
        return self._label(index)

    def __str__(self) -> str:
        return self.__class__.__name__
