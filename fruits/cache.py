from typing import List, Protocol, Dict

import numpy as np

from fruits._backend import _coquantile


class Cache(Protocol):
    """Protocol for classes that cache their calculation results of
    a time series dataset and using a key for accessing the results.
    """

    cache: Dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        """Returns the cached results for the given key.

        :type key: str
        :rtype: np.ndarray
        """

    def process(self, X: np.ndarray, keys: List[str]) -> None:
        """Processes the given time series dataset and caches the
        results.

        :type X: np.ndarray
        :param keys: Keys on which the results are calculated.
        :type keys: List[str]
        """


class CoquantileCache:
    """Class that matches the :class:`~fruits.cache.Cache` protocol and
    calculates coquantiles that are needed for a lot of transformations
    in a :class:`~fruits.core.fruit.Fruit`.
    """

    cache: Dict[str, np.ndarray]

    def __init__(self):
        self.cache = dict()

    def get(self, key: str) -> np.ndarray:
        """Returns the cached coquantiles.

        :rtype: np.ndarray
        """
        return self.cache[key]

    def process(self, X: np.ndarray, keys: List[str]) -> None:
        """Processes the given time series dataset and caches the
        resulting coquantiles for the given keys.

        :type X: np.ndarray
        :type keys: List[str]
        """
        for key in keys:
            self.cache[key] = _coquantile(X, float(key))

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)
