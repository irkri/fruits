from typing import Protocol

import numpy as np

from fruits._backend import _coquantile


class Cache(Protocol):
    """Protocol for classes that cache their calculation results of
    a time series dataset and use a key for accessing the results.
    """

    cache: dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        ...

    def process(self, X: np.ndarray, keys: list[str]) -> None:
        ...


class CoquantileCache:
    """Class that matches the :class:`~fruits.cache.Cache` protocol and
    calculates coquantiles that are needed for a lot of transformations
    in a :class:`~fruits.core.fruit.Fruit`.
    """

    cache: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.cache = dict()

    def get(self, key: str) -> np.ndarray:
        """Returns cached coquantiles for the given key."""
        return self.cache[key]

    def process(self, X: np.ndarray, keys: list[str]) -> None:
        """Processes the given time series dataset and caches the
        resulting coquantiles for the given keys.
        """
        for key in keys:
            self.cache[key] = _coquantile(X, float(key))

    def __getitem__(self, key: str) -> np.ndarray:
        return self.get(key)
