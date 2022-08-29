from abc import ABC, abstractmethod
from typing import TypeVar
import warnings

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


def force_input_shape(X: np.ndarray) -> np.ndarray:
    """Makes the attempt to format the input shape of the given
    multidimensional time series dataset.
    This leads to an three dimensional array where

    - ``X.shape[0]``: Number of time series
    - ``X.shape[1]``: Number of dimensions in each time series
    - ``X.shape[2]``: Length of each time series

    Raises:
        ValueError: If ``X.ndim > 3``.
    """
    if X.ndim < 3:
        warnings.warn("The input shape of the time series dataset "
                      "is < 3 and needs to be converted.")
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        return X
    elif X.ndim == 3:
        return X.copy()
    else:
        raise ValueError("Unsupported input shape of the given time series "
                         "dataset")
