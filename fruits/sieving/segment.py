__all__ = ["MAX", "MIN", "END", "CUR"]

from abc import ABC
from collections.abc import Sequence
from typing import Optional, Union

import numba
import numpy as np

from ..cache import CacheType, _increments
from .abstract import FeatureSieve


class SegmentSieve(FeatureSieve, ABC):
    """Abstract class that calculates coquantiles and quantiles of the
    input time series and evaluates the given sieve on the truncated
    input.

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
            quantile is calculated in these cases. Defaults to ``-1,1``.
    """

    def __init__(
        self,
        cut: Union[Sequence[float], float] = -1,
        q: Optional[Sequence[float]] = None,
    ) -> None:
        self._cut = cut if isinstance(cut, Sequence) else (cut, )
        self._q = q if isinstance(q, Sequence) else (-1.0, 1.0)

    @property
    def requires_fitting(self) -> bool:
        for q in self._q:
            if q not in [-1, 0, 1]:
                return True
        return False

    def _get_transformed_cuts(self, X: np.ndarray) -> np.ndarray:
        new_cuts = np.zeros((X.shape[0], len(self._cut)+1))
        for i, cut in enumerate(self._cut):
            if isinstance(cut, float):
                new_cuts[:, i+1] = self._cache.get(
                    CacheType.COQUANTILE,
                    str(cut),
                )
            else:
                new_cuts[:, i+1] = self._cut[i] if self._cut[i] >= 0 else (
                    X.shape[1] + self._cut[i] + 1
                )
        new_cuts = np.sort(new_cuts)
        return new_cuts.astype(np.int64)

    def _fit(self, X: np.ndarray) -> None:
        self._quantiles = np.zeros(len(self._q))
        for i, q in enumerate(self._q):
            if q == 1.0:
                self._quantiles[i] = np.inf
            elif q == -1.0:
                self._quantiles[i] = np.NINF
            elif q != 0:
                self._quantiles[i] = np.quantile(X, q)
        self._quantiles = np.sort(self._quantiles)

    def _get_unfitted_quantiles(self):
        self._quantiles = np.zeros(len(self._q))
        for i, q in enumerate(self._q):
            if q == 1.0:
                self._quantiles[i] = np.inf
            elif q == -1.0:
                self._quantiles[i] = np.NINF
            elif q != 0:
                raise RuntimeError("Sieve has not been fitted properly")

    def _nfeatures(self) -> int:
        return len(self._cut) * (len(self._q) - 1)

    def _copy(self):
        return self.__class__(self._cut, self._q)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._cut}, {self._q})"

    def _summary(self) -> str:
        string = f"{self.__class__.__name__} -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string


class MAX(SegmentSieve):
    """FeatureSieve: Maximal value

    This sieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
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
                    if cuts[i, j] == cuts[i, j+1]:
                        result[i, j*(quantiles.shape[0]-1)+k] = 0
                    else:
                        arr = X[i, cuts[i, j]:cuts[i, j+1]]
                        arr = arr[np.logical_and(
                            quantiles[k] < arr, arr <= quantiles[k+1]
                        )]
                        result[i, j*(quantiles.shape[0]-1)+k] = np.max(arr)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.requires_fitting:
            self._get_unfitted_quantiles()
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MAX._backend(X, cuts, self._quantiles)

    def _summary(self) -> str:
        string = f"MAX -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string


class MIN(SegmentSieve):
    """FeatureSieve: Minimal value

    This sieve returns the minimal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
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
                    if cuts[i, j] == cuts[i, j+1]:
                        result[i, j*(quantiles.shape[0]-1)+k] = 0
                    else:
                        arr = X[i, cuts[i, j]:cuts[i, j+1]]
                        arr = arr[np.logical_and(
                            quantiles[k] < arr, arr <= quantiles[k+1]
                        )]
                        result[i, j*(quantiles.shape[0]-1)+k] = np.min(arr)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.requires_fitting:
            self._get_unfitted_quantiles()
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MIN._backend(X, cuts, self._quantiles)

    def _summary(self) -> str:
        string = f"MIN -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string


class END(SegmentSieve):
    """FeatureSieve: Last value

    This FeatureSieve returns the last value of each time series in a
    given dataset.
    """

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        for j in range(cuts.shape[1] - 1):
            result[:, j] = np.take_along_axis(
                X,
                cuts[:, j+1:j+2]-1,
                axis=1
            )[:, 0]
        return result

    def _summary(self) -> str:
        string = f"END -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string


class CUR(SegmentSieve):
    """FeatureSieve: Curvature

    Calculates the curvature of the time series as the total sum of
    squared second-order increments.
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
        X_inc = _increments(
            _increments(np.expand_dims(X, axis=1), 1), 1,
        )[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                for k in range(quantiles.shape[0] - 1):
                    arr = X_inc[i, cuts[i, j]:cuts[i, j+1]]
                    arr = arr[np.logical_and(
                        quantiles[k] < arr, arr <= quantiles[k+1]
                    )]
                    result[i, j*(quantiles.shape[0]-1)+k] = np.sum(arr**2)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.requires_fitting:
            self._get_unfitted_quantiles()
        cuts = self._get_transformed_cuts(X, **kwargs)
        return CUR._backend(X, cuts, self._quantiles)

    def _summary(self) -> str:
        string = f"CUR -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string
