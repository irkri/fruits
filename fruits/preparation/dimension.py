__all__ = ["ONE", "DIM", "FFN"]

from typing import Any, Callable, Optional

import numba
import numpy as np

from .abstract import Preparateur


class ONE(Preparateur):
    """Preparateur: Ones

    Preparateur that appends a dimension to each time series consisting
    of only ones.
    """

    @property
    def requires_fitting(self) -> bool:
        return False

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

    @property
    def requires_fitting(self) -> bool:
        return False

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


class FFN(Preparateur):
    """Preparateur: Feed-Forward Two-Layer Neural Network

    Transforms single values of a time series. A neural network with one
    hidden layer and a ReLU activation function is used. All weights and
    biases are gaussian distributed with mean zero.

    Args:
        n (int, optional): Number of nodes in the hidden layer. Defaults
            to 10.
        dim (int, optional): Which dimension in the input time series to
            transform. Defaults to all dimensions.
        center (bool, optional): Whether to center the time series
            before doing the transformation. This will most likely lead
            to better results because of the involved ReLU operation.
            Defaults to true.
        std (float, optional): Standard deviation of the gaussian
            distributed weights and biases used. Defaults to 1.
        overwrite (bool, optional): If set to true, the preparateur will
            replace the original dimension with the new one. Otherwise a
            new dimensions gets appended. Defaults to false.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:]("
            "float64[:,:,:], float64[:], float64[:], float64[:], boolean)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(
        X: np.ndarray,
        weights1: np.ndarray,
        biases1: np.ndarray,
        weights2: np.ndarray,
        center: bool,
    ) -> np.ndarray:
        result = X.copy()
        if center:
            for i in numba.prange(X.shape[0]):
                for j in numba.prange(X.shape[1]):
                    c = np.mean(X[i, j])
                    for k in numba.prange(X.shape[2]):
                        result[i, j, k] -= c
        for i in numba.prange(X.shape[0]):
            for j in numba.prange(X.shape[1]):
                for k in numba.prange(X.shape[2]):
                    layer1 = weights1 * X[i, j, k] + biases1
                    layer1 = (layer1 * (layer1 > 0))
                    result[i, j, k] = np.sum(weights2 * layer1)
        return result

    def __init__(
        self,
        n: int = 10,
        dim: Optional[int] = None,
        center: bool = True,
        std: float = 1.0,
        overwrite: bool = False,
    ) -> None:
        self._n = n
        self._dim = dim
        self._center = center
        self._std = std
        self._overwrite = overwrite

    def _fit(self, X: np.ndarray) -> None:
        self._weights1 = np.random.normal(scale=self._std, size=self._n)
        self._biases1 = np.random.normal(scale=self._std, size=self._n)
        self._weights2 = np.random.normal(scale=self._std, size=self._n)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_weights1"):
            raise RuntimeError("Preparateur FFN was not fitted")

        new_dim = X if self._dim is None else X[:, self._dim:self._dim+1, :]

        new_dim = FFN._backend(
            new_dim,
            self._weights1,
            self._biases1,
            self._weights2,
            self._center,
        )

        if self._overwrite:
            if self._dim is None:
                result = new_dim
            else:
                result = X.copy()
                result[:, self._dim, :] = new_dim[:, 0, :]
        else:
            if self._dim is None:
                result = np.zeros((X.shape[0], 2*X.shape[1], X.shape[2]))
                result[:, :X.shape[1], :] = X.copy()
                result[:, X.shape[1]:, :] = new_dim
            else:
                result = np.zeros((X.shape[0], X.shape[1]+1, X.shape[2]))
                result[:, :X.shape[1], :] = X.copy()
                result[:, X.shape[1], :] = new_dim[:, 0, :]
        return result

    def _copy(self) -> "FFN":
        return FFN(
            self._n,
            self._dim,
            self._center,
            self._std,
            self._overwrite,
        )

    def __str__(self) -> str:
        return (f"FFN({self._n}, {self._dim}, {self._center}, {self._std}, "
                f"{self._overwrite})")
