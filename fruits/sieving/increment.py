__all__ = ["NPI", "MPI", "XPI", "LPI"]

from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numba
import numpy as np

from ..cache import _increments
from .segment import SegmentSieve


class IncrementSieve(SegmentSieve):
    """Abstract sieve that calculates coquantiles and quantiles of the
    input time series or increments thereof and evaluates the given
    sieve on the truncated input.

    Args:
        cut (int | float or Sequence, optional): If ``cut`` is an index
            in the time series ``X`` array, the features are sieved from
            ``X[:cut]``. If it is a float in ``[0,1]``, the
            corresponding 'coquantile' will be calculated first. This
            argument can also be a list of floats or integers which will
            be treated individually the same way. Defaults to ``1``.
        q (Sequence of float, optional): A probability in ``[0, 1]`` or
            ``-1`` for which the corresponding quantile of the time
            series dataset gets calculated (for each ``cut``). The value
            ``1.0`` is referring to ``numpy.inf``, ``-1.0`` to
            ``numpy.ninf`` and ``0.0`` is actually the value ``0``, no
            quantile is calculated in these cases. Defaults to ``0,1``.
        inc (int, optional): Depth of the increments to calculate before
            applying any transformation. A positive number evaluates the
            sieve on the input after calculating ``inc``-times its
            increments. A negative number evaluates the sieve on
            ``-inc``-times its cumulative sum, i.e. the inverse of the
            increments. Defaults to 1.
    """

    @staticmethod
    @abstractmethod
    def _backend(
        X: np.ndarray,
        cuts: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        ...

    def __init__(
        self,
        cut: Union[Sequence[float], float] = -1,
        q: Optional[Sequence[float]] = None,
        inc: int = 1,
    ) -> None:
        super().__init__(cut, q if q is not None else (0.0, 1.0))
        self._inc = inc

    def _pre_transform(self, X: np.ndarray) -> np.ndarray:
        arr = X.copy()
        if self._inc > 0:
            for _ in range(self._inc):
                arr = _increments(arr[:, np.newaxis, :], 1)[:, 0, :]
        elif self._inc < 0:
            for _ in range(-self._inc):
                arr = np.cumsum(arr, axis=1)
        return arr

    def _fit(self, X: np.ndarray) -> None:
        super()._fit(self._pre_transform(X))

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.requires_fitting:
            self._get_unfitted_quantiles()
        arr = self._pre_transform(X)
        cuts = self._get_transformed_cuts(arr, **kwargs)
        return self.__class__._backend(arr, cuts, self._quantiles)

    def _copy(self):
        return self.__class__(self._cut, self._q, self._inc)

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"({self._cut}, {self._q}, {self._inc})")

    def _summary(self) -> str:
        string = f"{self.__class__.__name__} -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string


class NPI(IncrementSieve):
    """FeatureSieve: Number of Positive Increments

    Counts the number of positive increments in the given time series.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:], float64[:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(
        X: np.ndarray,
        cuts: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        nfeatures = (cuts.shape[1] - 1) * (quantiles.shape[0] - 1)
        result = np.zeros((X.shape[0], nfeatures))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                for k in range(quantiles.shape[0] - 1):
                    arr = X[i, cuts[i, j]:cuts[i, j+1]]
                    arr = np.logical_and(
                        quantiles[k] < arr, arr <= quantiles[k+1]
                    )
                    result[i, j*(quantiles.shape[0]-1)+k] = np.sum(arr)
        return result


class MPI(IncrementSieve):
    """FeatureSieve: Mean of Positive Increments

    Returns the mean of the increasing part of the given time series.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:], float64[:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(
        X: np.ndarray,
        cuts: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        nfeatures = (cuts.shape[1] - 1) * (quantiles.shape[0] - 1)
        result = np.zeros((X.shape[0], nfeatures))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                for k in range(quantiles.shape[0] - 1):
                    arr = X[i, cuts[i, j]:cuts[i, j+1]]
                    arr = arr[np.logical_and(
                        quantiles[k] < arr, arr <= quantiles[k+1]
                    )]
                    if arr.size == 0:
                        result[i, j*(quantiles.shape[0]-1)+k] = 0
                    else:
                        result[i, j*(quantiles.shape[0]-1)+k] = np.mean(arr)
        return result


class XPI(IncrementSieve):
    """FeatureSieve: Mean of Indices of Positive Increments

    Returns the mean of indices where the given time series is
    increasing.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:], float64[:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(
        X: np.ndarray,
        cuts: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        nfeatures = (cuts.shape[1] - 1) * (quantiles.shape[0] - 1)
        result = np.zeros((X.shape[0], nfeatures))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                for k in range(quantiles.shape[0] - 1):
                    arr = X[i, cuts[i, j]:cuts[i, j+1]]
                    ind = np.where(np.logical_and(
                        quantiles[k] < arr, arr <= quantiles[k+1]
                    ))[0]
                    if ind.size == 0:
                        result[i, j*(quantiles.shape[0]-1)+k] = 0
                    else:
                        result[i, j*(quantiles.shape[0]-1)+k] = np.mean(ind)
        return result


class LPI(IncrementSieve):
    """FeatureSieve: Longest Slice of Positive Increments

    Returns the length of the longest time span where the time series is
    increasing.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:], float64[:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(
        X: np.ndarray,
        cuts: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        nfeatures = (cuts.shape[1] - 1) * (quantiles.shape[0] - 1)
        result = np.zeros((X.shape[0], nfeatures))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                for k in range(quantiles.shape[0] - 1):
                    arr = X[i, cuts[i, j]:cuts[i, j+1]]
                    arr = np.logical_and(
                        quantiles[k] < arr, arr <= quantiles[k+1]
                    )
                    longest = 0
                    current = 0
                    for s in range(arr.shape[0]):
                        if arr[s]:
                            current += 1
                            if current > longest:
                                longest = current
                        else:
                            current = 0
                    result[i, j*(quantiles.shape[0]-1)+k] = longest
        return result
