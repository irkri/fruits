from abc import ABC, abstractmethod
from typing import Literal, Sequence

import numpy as np

from .words import Word
from ..cache import CacheType, SharedSeedCache


class Weighting(ABC):

    _cache: SharedSeedCache

    @abstractmethod
    def __init__(self, word: Word) -> None:
        ...

    @abstractmethod
    def weights(self, n: int, k: int, i: int) -> np.ndarray:
        ...

    def get_fast_args(self, n: int, l: int) -> np.ndarray:
        raise NotImplementedError(
            "Weighting not supported for compiled calculation of iterated sums"
        )


class ExponentialWeighting(Weighting):
    """Exponential penalization for the calculation of iterated sums.
    Sums that use multiplications of time steps that are further apart
    from each other are scaled down exponentially. For two time steps
    ``i`` and ``j`` in the iterated sum, the summand is scaled by::

        e^(a*(j-i-1))

    where ``a`` is a given scalar. This scalar can be specified in a
    list of floats, each single float being applied to two consecutive
    indices for consecutive extended letters in words used by the
    iterated sum. An appropriate number of scalars have to be specified,
    matching or exceeding the length of the longest word in the
    :class:`ISS`.

    Args:
        scalars (sequence of float): The float values used to scale the
            time index differences.
        lookup ("indices", "L1" or "L2"): An array where instead of
            indices, appropriate values are used to compute the time
            index differences. For example, "L1" refers to the sum of
            absolute values of increments. "L2" is the sum of squared
            increments up to the given time step. Defaults to "indices".
    """

    def __init__(
        self,
        scalars: Sequence[float],
        lookup: Literal["indices", "L1", "L2"] = "indices",
    ) -> None:
        self._scalars = np.array(
            [.0]+list(scalars)+[.0],
            dtype=np.float32,
        )
        self._lookup = lookup

    def weights(self, n: int, k: int, i: int) -> np.ndarray:
        if self._lookup != "indices":
            lookup = self._cache.get(CacheType.ISS, self._lookup)[i]
        else:
            lookup = np.arange(n)
        return np.exp(
            lookup * (self._scalars[k+1] - self._scalars[k])
        )

    def get_fast_args(self, n: int, l: int) -> tuple[np.ndarray, np.ndarray]:
        if self._lookup != "indices":
            lookup = self._cache.get(CacheType.ISS, self._lookup)
        else:
            lookup = np.ones((n, l)) * np.arange(l)
        return self._scalars, lookup
