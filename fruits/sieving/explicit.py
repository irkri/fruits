__all__ = ["MAX", "MIN", "END", "NPI", "MPI", "XPI", "LPI", "CUR"]

from abc import ABC
from collections.abc import Sequence
from typing import Union

import numba
import numpy as np

from ..cache import CacheType, _increments
from .abstract import FeatureSieve


class ExplicitSieve(FeatureSieve, ABC):
    """Abstract class that calculates coquantiles of the input time
    series and evaluates the given sieve on the truncated input.

    Args:
        cut (int | float or Sequence): If ``cut`` is an index in the
            time series ``X`` array, the features are sieved from
            ``X[:cut]``. If it is a float in ``[0,1]``, the
            corresponding 'coquantile' will be calculated first. This
            argument can also be a list of floats or integers which will
            be treated individually the same way.
    """

    def __init__(
        self,
        cut: Union[Sequence[float], float] = -1,
    ) -> None:
        self._cut = cut if isinstance(cut, Sequence) else (cut, )

    @property
    def requires_fitting(self) -> bool:
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

    def _nfeatures(self) -> int:
        return len(self._cut)


class MAX(ExplicitSieve):
    """FeatureSieve: Maximal value

    This sieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                if cuts[i, j] == cuts[i, j+1]:
                    result[i, j] = 0
                else:
                    result[i, j] = np.max(X[i, cuts[i, j]:cuts[i, j+1]])
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MAX._backend(X, cuts)

    def _summary(self) -> str:
        string = f"MAX -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "MAX":
        return MAX(self._cut)

    def __str__(self) -> str:
        return f"MAX({self._cut})"


class MIN(ExplicitSieve):
    """FeatureSieve: Minimal value

    This sieve returns the minimal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                if cuts[i, j] == cuts[i, j+1]:
                    result[i, j] = 0
                else:
                    result[i, j] = np.min(X[i, cuts[i, j]:cuts[i, j+1]])
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MIN._backend(X, cuts)

    def _summary(self) -> str:
        string = f"MIN -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "MIN":
        return MIN(self._cut)

    def __str__(self) -> str:
        return f"MIN({self._cut})"


class END(ExplicitSieve):
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

    def _copy(self) -> "END":
        return END(self._cut)

    def __str__(self) -> str:
        return f"END({self._cut})"


class NPI(ExplicitSieve):
    """FeatureSieve: Number of Positive Increments

    Counts the number of positive increments in the given time series.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        X_inc = _increments(np.expand_dims(X, axis=1), 1)[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                result[i, j] = np.sum(X_inc[i, cuts[i, j]:cuts[i, j+1]] > 0)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return NPI._backend(X, cuts)

    def _summary(self) -> str:
        string = f"NPI -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "NPI":
        return NPI(self._cut)

    def __str__(self) -> str:
        return f"NPI({self._cut})"


class MPI(ExplicitSieve):
    """FeatureSieve: Mean of Positive Increments

    Returns the mean of the increasing part of the given time series.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        X_inc = _increments(np.expand_dims(X, axis=1), 1)[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                part = X_inc[i, cuts[i, j]:cuts[i, j+1]]
                part = part[part > 0]
                if part.size == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = np.mean(part)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MPI._backend(X, cuts)

    def _summary(self) -> str:
        string = f"MPI -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "MPI":
        return MPI(self._cut)

    def __str__(self) -> str:
        return f"MIA({self._cut})"


class XPI(ExplicitSieve):
    """FeatureSieve: Mean of Indices of Positive Increments

    Returns the mean of indices where the given time series is
    increasing.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        X_inc = _increments(np.expand_dims(X, axis=1), 1)[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                indices = np.where(X_inc[i, cuts[i, j]:cuts[i, j+1]] > 0)[0]
                if indices.size == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = np.mean(indices)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return XPI._backend(X, cuts)

    def _summary(self) -> str:
        string = f"XPI -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "XPI":
        return XPI(self._cut)

    def __str__(self) -> str:
        return f"XPI({self._cut})"


class LPI(ExplicitSieve):
    """FeatureSieve: Longest Slice of Positive Increments

    Returns the length of the longest time span where the time series is
    increasing.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1), dtype=np.float64)
        X_inc = _increments(np.expand_dims(X, axis=1), 1)[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                part = X_inc[i, cuts[i, j]:cuts[i, j+1]] > 0
                longest = 0
                current = 0
                for k in range(part.shape[0]):
                    if part[k]:
                        current += 1
                        if current > longest:
                            longest = current
                    else:
                        current = 0
                result[i, j] = longest
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return LPI._backend(X, cuts)

    def _summary(self) -> str:
        string = f"LPI -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "LPI":
        return LPI(self._cut)

    def __str__(self) -> str:
        return f"LPI({self._cut})"


class CUR(ExplicitSieve):
    """FeatureSieve: Curvature

    Calculates the curvature of the time series as the total sum of
    squared second-order increments.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        fastmath=True,
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        X_inc = _increments(
            _increments(np.expand_dims(X, axis=1), 1), 1,
        )[:, 0, :]
        for i in numba.prange(X.shape[0]):
            for j in range(cuts.shape[1] - 1):
                result[i, j] = np.sum(X_inc[i, cuts[i, j]:cuts[i, j+1]]**2)
        return result

    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return CUR._backend(X, cuts)

    def _summary(self) -> str:
        string = f"CUR -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def _copy(self) -> "CUR":
        return CUR(self._cut)

    def __str__(self) -> str:
        return f"CUR({self._cut})"
