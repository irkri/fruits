from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

TCopy = TypeVar("TCopy", bound="Seed")


class Seed(ABC):
    """Abstract class for all additional objects that can be added to a
    fruit including preparateurs, words and sieves.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the seed to the given data."""

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transforms the given data and returns the results."""

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Equivalent of calling ``fit`` and ``transform`` consecutively
        on this object.
        """
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    @abstractmethod
    def _copy(self: TCopy) -> TCopy:
        ...

    def copy(self: TCopy) -> TCopy:
        """Returns a copy of this seed."""
        return self._copy()

    def __str__(self) -> str:
        return self.__class__.__name__
