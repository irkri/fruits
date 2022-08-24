from typing import Callable

import numpy as np

from fruits.preparation.abstract import DataPreparateur


class ONE(DataPreparateur):
    """DataPreparateur: Ones

    Preparateur that appends a dimension to each time series consisting
    of only ones.
    """

    def __init__(self) -> None:
        super().__init__("One")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X_new = np.ones((X.shape[0], X.shape[1]+1, X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        return X_new

    def copy(self) -> "ONE":
        return ONE()

    def __eq__(self, other) -> bool:
        return True

    def __str__(self) -> str:
        return "ONE()"

    def __repr__(self) -> str:
        return "fruits.preparation.dimension.ONE"


class DIM(DataPreparateur):
    """DataPreparateur: Dimension Creator

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
        super().__init__("Dimension Creator")
        self._function = f

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        new_dims = self._function(X)
        X_new = np.zeros((X.shape[0],
                          X.shape[1] + new_dims.shape[1],
                          X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        X_new[:, X.shape[1]:, :] = new_dims[:, :, :]
        return X_new

    def copy(self) -> "DIM":
        return DIM(self._function)

    def __eq__(self, other) -> bool:
        return False

    def __str__(self) -> str:
        return f"DIM(f={self._function.__name__})"

    def __repr__(self) -> str:
        return "fruits.preparation.dimension.DIM"
