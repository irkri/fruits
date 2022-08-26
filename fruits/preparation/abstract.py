from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DataPreparateur(ABC):
    """Abstract class for a data preparateur.

    A preparateur can be fitted on a three dimensional numpy array
    (preferably containing time series data). The output of
    ``self.transform`` is a numpy array that matches the shape of the
    input array.
    A class derived from DataPreparateur can be added to a
    ``fruits.Fruit`` object for the preprocessing step.

    Args:
        name (str, optional): Identification string of the feature
            sieve. Defaults to an empty string.
    """

    def _fit(self, X: np.ndarray, **kwargs) -> None:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the preparateur to the given dataset."""
        self._fit(X, **kwargs)

    @abstractmethod
    def _transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transforms the given timeseries dataset."""
        return self._transform(X, **kwargs)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Equivalent of calling ``DataPreparateur.fit`` and
        ``DataPreparateur.transform`` consecutively.
        """
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    def _get_cache_keys(self) -> dict[str, list[str]]:
        # returns keys for cache needed in the sieve
        return dict()

    @abstractmethod
    def _copy(self) -> "DataPreparateur":
        ...

    def copy(self) -> "DataPreparateur":
        """Returns a copy of this preparateur."""
        return self._copy()

    def __eq__(self, other: Any) -> bool:
        return False
