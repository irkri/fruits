from abc import ABC, abstractmethod

import numpy as np

from ..seed import Seed


class FeatureSieve(Seed, ABC):
    """Abstract class for a feature sieve. Sieves are the last part of a
    :class:`~fruits.core.fruit.Fruit`.

    A feature sieve is used to transforms a two-dimensional numpy
    array containing iterated sums into a one dimensional numpy array of
    features.
    """

    @abstractmethod
    def _nfeatures(self) -> int:
        ...

    def nfeatures(self) -> int:
        """Returns the number of features this sieve produces."""
        return self._nfeatures()

    def _fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def _summary(self) -> str:
        ...

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        return self._summary()
