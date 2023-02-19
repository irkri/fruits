from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .words import Word


class Weighting(ABC):

    @abstractmethod
    def __init__(self, word: Word) -> None:
        ...

    @abstractmethod
    def weights(self, n: int, k: int) -> np.ndarray:
        ...

    def get_fast_args(self) -> np.ndarray:
        raise NotImplementedError(
            "Weighting not supported for compiled calculation of iterated sums"
        )


class ExponentialWeighting(Weighting):
    """Exponential penalization for the calculation of iterated sums.
    Sums that use multiplications of time steps that are further apart
    from each other are scaled down exponentially. For two time steps
    ``i`` and ``j`` in the iterated sum, the summand is scaled by::

        e^(a*(j-i-1))

    where a is a given scalar. This scalar can be specified in a list of
    floats, each single float being applied to two consecutive indices
    for consecutive extended letters in words used by the iterated sum.
    An appropriate number of scalars have to be specified, matching or
    exceeding the length of the longest word in the :class:`ISS`.
    """

    def __init__(self, scalars: Sequence[float]) -> None:
        self._scalars = np.array(
            [.0]+list(scalars)+[.0],
            dtype=np.float32,
        )

    def weights(self, n: int, k: int) -> np.ndarray:
        return np.exp(
            np.arange(n) * (self._scalars[k+1]-self._scalars[k])
            + self._scalars[k]
        )

    def get_fast_args(self) -> np.ndarray:
        return self._scalars
