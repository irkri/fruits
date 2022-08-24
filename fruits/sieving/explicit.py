from abc import ABC
from typing import Union

import numba
import numpy as np

from fruits.sieving.abstract import FeatureSieve
from fruits._backend import _increments
from fruits.cache import CoquantileCache


class ExplicitSieve(FeatureSieve, ABC):
    """Abstract class that has the ability to calculate cutting points
    as indices in the time series based on a given 'coquantile'.
    A (non-scaled) value returned by an explicit sieve always also is a
    value in the original time series.

    Args:
        cut (int/float): If ``cut`` is an index in the time series
            array, the features are sieved from ``X[:cut]``. If it is a
            float in ``[0,1]``, the corresponding 'coquantile' will be
            calculated first. This option can also be a list of floats
            or integers which will be treated the same way. The default
            is sieving from the whole time series.
        segments (bool, optional): If set to ``True``, then the cutting
            indices will be sorted and treated as interval borders and
            the maximum in each interval will be sieved. The left
            interval border is reduced by 1 before slicing. This means
            that an input of ``cut=[1,5,10]`` results in two features
            ``max(X[0:5])`` and ``max(X[4:10])``.
            If set to ``False``, then the left interval border is always
            0. Defaults to ``False``.
    """

    _CACHE_KEY = 'coquantile'

    def __init__(
        self,
        cut: Union[list[float], float] = -1,
        segments: bool = False,
        name: str = "Abstract Explicit Sieve",
    ) -> None:
        super().__init__(name)
        self._cut = cut if isinstance(cut, list) else [cut]
        if len(self._cut) == 1 and segments:
            self._cut = [1, self._cut[0]]
        self._segments = segments

    def _get_cache_keys(self) -> dict[str, list[str]]:
        # transforms the input cuts based on the given time series
        keys = [str(cut) for cut in self._cut if isinstance(cut, float)]
        return {self._CACHE_KEY: keys}

    def _get_cache(self, X: np.ndarray, **kwargs) -> CoquantileCache:
        # checks if the sieve got any cache to use, if not, compute it
        if 'cache' in kwargs:
            return kwargs['cache']
        else:
            cache = CoquantileCache()
            cache.process(np.expand_dims(X, axis=1),
                          [str(cut) for cut in self._cut
                           if isinstance(cut, float)])
            return cache

    def _get_transformed_cuts(self, X: np.ndarray, **kwargs) -> np.ndarray:
        # mix cuts got from coquantile cache with given integer cuts
        cached_cuts = self._get_cache(X, **kwargs)
        new_cuts = np.zeros((X.shape[0], len(self._cut)))
        for i, cut in enumerate(self._cut):
            if isinstance(cut, float):
                new_cuts[:, i] = cached_cuts[str(cut)]
            else:
                if self._cut[i] <= 0:
                    new_cuts[:, i] = X.shape[1] + self._cut[i] + 1
                else:
                    new_cuts[:, i] = self._cut[i]
        if self._segments:
            new_cuts = np.sort(new_cuts)
        return new_cuts.astype(np.int64)

    def nfeatures(self) -> int:
        """Returns the number of features this sieve produces."""
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)


class MAX(ExplicitSieve):
    """FeatureSieve: Maximal value

    This sieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    """

    def __init__(self, cut: Union[list[float], float] = -1) -> None:
        super().__init__(cut, True, "Maximal value")

    @staticmethod
    @numba.njit(
        "float64[:,:](float64[:,:], int64[:,:])",
        parallel=True,
        cache=True,
    )
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        for i in numba.prange(X.shape[0]):  # pylint: disable=not-an-iterable
            for j in range(1, cuts.shape[1]):
                result[i, j-1] = np.max(X[i, cuts[i, j-1]-1:cuts[i, j]])
        return result

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MAX._backend(X, cuts)

    def summary(self) -> str:
        string = "MAX"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "MAX":
        fs = MAX(self._cut)
        return fs

    def __str__(self) -> str:
        string = "MAX(" + \
                f"cut={self._cut})"
        return string


class MIN(ExplicitSieve):
    """FeatureSieve: Minimal value

    This sieve returns the minimal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    """

    def __init__(self, cut: Union[list[float], float] = -1) -> None:
        super().__init__(cut, True, "Minimum value")

    @staticmethod
    @numba.njit("float64[:,:](float64[:,:], int64[:,:])",
                parallel=True, cache=True)
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1] - 1))
        for i in numba.prange(X.shape[0]):  # pylint: disable=not-an-iterable
            for j in range(1, cuts.shape[1]):
                result[i, j-1] = np.min(X[i, cuts[i, j-1]-1:cuts[i, j]])
        return result

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return MIN._backend(X, cuts)

    def summary(self) -> str:
        string = "MIN"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "MIN":
        fs = MIN(self._cut)
        return fs

    def __str__(self) -> str:
        string = "MIN(" + \
                f"cut={self._cut})"
        return string


class END(ExplicitSieve):
    """FeatureSieve: Last value

    This FeatureSieve returns the last value of each time series in a
    given dataset.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    The option 'segments' will be ignored in this sieve.
    """

    def __init__(self, cut: Union[list[float], float] = -1) -> None:
        super().__init__(cut, False, "Last value")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        result = np.zeros((X.shape[0], self.nfeatures()))
        for j in range(cuts.shape[1]):
            result[:, j] = np.take_along_axis(
                X,
                cuts[:, j:j+1]-1,
                axis=1
            )[:, 0]
        return result

    def summary(self) -> str:
        string = f"END -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "END":
        fs = END(self._cut)
        return fs

    def __str__(self) -> str:
        string = "END(" + \
                f"cut={self._cut})"
        return string


class PIA(ExplicitSieve):
    """FeatureSieve: Proportion of incremental alteration

    Counts the number of positive changes in the given time series. This
    is equal to the number of values greater than zero in the increments
    of the time series. This number will be divided (by default) by the
    length of the time series.
    With 'segments' set to ``False``, this sieve is time warping
    invariant.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    """

    def __init__(self, cut: Union[list[float], float] = -1) -> None:
        super().__init__(cut, False, "Proportion of incremental alteration")

    @staticmethod
    @numba.njit("float64[:,:](float64[:,:], int64[:,:])",
                parallel=True, cache=True)
    def _backend(X: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        result = np.zeros((X.shape[0], cuts.shape[1]))
        X_inc = _increments(np.expand_dims(X, axis=1))[:, 0, :]
        for i in numba.prange(X.shape[0]):  # pylint: disable=not-an-iterable
            for j in range(cuts.shape[1]):
                result[i, j] = np.sum(X_inc[i, :cuts[i, j]] > 0)
        result[:, :] /= X.shape[1]
        return result

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        return PIA._backend(X, cuts)

    def summary(self) -> str:
        string = "PIA"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "PIA":
        fs = PIA(self._cut)
        return fs

    def __str__(self) -> str:
        string = "PIA(" + \
                f"cut={self._cut})"
        return string


class LCS(ExplicitSieve):
    """FeatureSieve: Length of coquantile slices

    Returns the length of coquantile slices of each given time series.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    """

    def __init__(
        self,
        cut: Union[list[float], float] = -1,
        segments: bool = False,
    ) -> None:
        super().__init__(cut, segments, "Length of coquantile slices")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cuts = self._get_transformed_cuts(X, **kwargs)
        result = np.zeros((X.shape[0], self.nfeatures()))
        if self._segments:
            for j in range(1, cuts.shape[1]):
                result[:, j-1] = cuts[:, j] - cuts[:, j-1] + 1
        else:
            for j in range(cuts.shape[1]):
                result[:, j] = cuts[:, j]
        return result

    def summary(self) -> str:
        string = "LCS"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "LCS":
        fs = LCS(self._cut, self._segments)
        return fs

    def __str__(self) -> str:
        string = "LCS(" + \
                f"cut={self._cut}, " + \
                f"segments={self._segments})"
        return string
