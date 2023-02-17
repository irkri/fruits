from abc import ABC, abstractmethod

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

    def __init__(self, word: Word) -> None:
        self._scalars = np.array([0.] + word.alpha + [0.], dtype=np.float32)

    def weights(self, n: int, k: int) -> np.ndarray:
        return np.exp(
            np.arange(n) * (self._scalars[k+1]-self._scalars[k])
            + self._scalars[k]
        )

    def get_fast_args(self) -> np.ndarray:
        return self._scalars
