from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..seed import Seed


class Preparateur(Seed, ABC):
    """Abstract class for a preparateur.

    A preparateur can be fitted on a three dimensional numpy array
    (preferably containing time series data). The output of
    ``self.transform`` is a numpy array that matches the shape of the
    input array.
    A class derived from ``Preparateur`` can be added to a
    ``fruits.Fruit`` object for the preprocessing step.

    Args:
        name (str, optional): Identification string of the feature
            sieve. Defaults to an empty string.
    """

    def _fit(self, X: np.ndarray, **kwargs) -> None:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        self._fit(X, **kwargs)

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._transform(X, **kwargs)

    def _get_cache_keys(self) -> dict[str, list[str]]:
        return dict()

    def __eq__(self, other: Any) -> bool:
        return False
