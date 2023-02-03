from enum import Enum, auto
from typing import Union, Optional

import numba
import numpy as np


@numba.njit("float64[:,:,:](float64[:,:,:])", cache=True)
def _increments(X: np.ndarray) -> np.ndarray:
    # calculates the increments of each time series in X
    result = np.zeros(X.shape)
    result[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]  # type: ignore
    return result


@numba.njit("int64[:](float64[:,:,:], float32)", parallel=False, cache=True)
def _coquantile(X: np.ndarray, q: float) -> np.ndarray:
    # calculates the coquantiles of each time series in X for given q
    Y = _increments(X)[:, 0, :]
    results = np.zeros(Y.shape[0], dtype=np.int64)
    for i in numba.prange(Y.shape[0]):
        Y[i, :] = np.cumsum(Y[i, :] * Y[i, :])
        results[i] = np.sum(Y[i, :] <= q * Y[i, -1])
    return results


class CacheType(Enum):
    """Defines different types of cache that are supplied to all seeds
    in a fruit. The cache is grouped in a :class:`SharedSeedCache`.
    """
    COQUANTILE = auto()


class SharedSeedCache:
    """Class that is given at a :meth:`fruits.Fruit.fit` call to every
    seed in the :class:`~fruits.Fruit`. It manages the cache that can be
    reused by all the different components like coquantiles.

    Args:
        X (np.ndarray): Input data for which all transformations will be
            applied.
    """

    def __init__(self, X: np.ndarray) -> None:
        self._cache: dict[CacheType, dict[str, Union[None, np.ndarray]]] = {
            CacheType.COQUANTILE: {}
        }
        self._input = X

    def get(
        self,
        cache_id: CacheType,
        key: str,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns a stored cache or calculates and stores the results
        of the target cache type.

        Args:
            cache_id (CacheType): Type of the calculation used.
            key (str): A key that can be used for different
                configurations of the given cache type.
            X (np.ndarray, optional): If supplied, calculates the cache
                with ``X`` as input if not already a stored value exists
                for the given key. Otherwise, the stored class variable
                ``X`` will be used. Defaults to None.
        """
        if (key not in self._cache[cache_id].keys()
                or self._cache[cache_id][key] is None):
            if X is None:
                self._cache[cache_id][key] = _coquantile(
                    self._input,
                    float(key),
                )
            else:
                self._cache[cache_id][key] = _coquantile(X, float(key))
        return self._cache[cache_id][key]  # type: ignore
