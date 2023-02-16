__all__ = ["ONE", "DIM", "LAY"]

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


class LAY(Preparateur):
    """Preparateur: Two-Layer Nerual Network

    Adds a dimension to the given time series dataset which is the
    linear combination of a transformed existing dimension.
    The transformation involves computing the pointwise ReLU of the
    linearly transformed input series.

    Args:
        n (int, optional): Number of linear transformations to use in
            the first layer of this two-step transform. Defaults to 10.
        dim (int, optional): The dimension to use from the input time
            series. Defaults to the first dimension.
        std (float, optional): Standard deviation of the normally
            distributed weights and biases in all linear transformations
            used. Defaults to 1.
        overwrite (bool, optional): If set to true, the preparateur will
            replace the original dimension with the new one.
    """

    def __init__(
        self,
        n: int = 10,
        dim: int = 0,
        std: float = 1.0,
        overwrite: bool = False,
    ) -> None:
        self._n = n
        self._dim = dim
        self._std = std
        self._overwrite = overwrite

    def _fit(self, X: np.ndarray) -> None:
        self._weights1 = np.random.normal(scale=self._std, size=self._n)
        self._biases1 = np.random.normal(scale=self._std, size=self._n)
        self._weights2 = np.random.normal(scale=self._std, size=self._n)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_weights1"):
            raise RuntimeError("Preparateur LAY was not fitted")
        new_dim = np.outer(self._weights1, X[:, self._dim, :]).reshape(
            self._n, X.shape[0], X.shape[2]
        ) + self._biases1[:, np.newaxis, np.newaxis]
        new_dim = new_dim * (new_dim > 0)
        new_dim = np.tensordot(self._weights2, new_dim, axes=1)
        if not self._overwrite:
            result = np.zeros((X.shape[0], X.shape[1]+1, X.shape[2]))
            result[:, :X.shape[1], :] = X
            result[:, X.shape[1], :] = new_dim
        else:
            result = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
            result[:, :, :] = X
            result[:, self._dim, :] = new_dim
        return result

    def _copy(self) -> "LAY":
        return LAY(self._n, self._dim, self._std)

    def __str__(self) -> str:
        return f"LAY({self._n}, {self._dim}, {self._std}, {self._overwrite})"
