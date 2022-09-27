__all__ = ["ONE", "DIM"]

from typing import Any, Callable

import numpy as np

from .abstract import Preparateur


class ONE(Preparateur):
    """Preparateur: Ones

    Preparateur that appends a dimension to each time series consisting
    of only ones.
    """

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_new = np.ones((X.shape[0], X.shape[1]+1, X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        return X_new

    def _copy(self) -> "ONE":
        return ONE()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ONE):
            return False
        return True

    def __str__(self) -> str:
        return "ONE()"


class DIM(Preparateur):
    """Preparateur: Dimension Creator

    Creates a new dimension in the given (multidimensional) time series
    dataset based on the supplied function.

    Args:
        f (Callable): Function that takes in a three dimensional numpy
            array of shape ``(n, d, l)`` and returns an array of shape
            ``(n, p, l)`` where ``p`` is an arbitrary integer matching
            the number of new dimensions that will be added to the input
            array.
    """

    def __init__(self, f: Callable[[np.ndarray], np.ndarray]) -> None:
        self._function = f

    def _transform(self, X: np.ndarray) -> np.ndarray:
        new_dims = self._function(X)
        X_new = np.zeros((X.shape[0],
                          X.shape[1] + new_dims.shape[1],
                          X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        X_new[:, X.shape[1]:, :] = new_dims[:, :, :]
        return X_new

    def _copy(self) -> "DIM":
        return DIM(self._function)

    def __str__(self) -> str:
        return f"DIM(f={self._function.__name__})"
