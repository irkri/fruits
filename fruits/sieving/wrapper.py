__all__ = ["INC"]

import numpy as np

from ..cache import _increments
from .abstract import FeatureSieve


class INC(FeatureSieve):
    """Wrapper for another feature sieve. INC will evaluate the given
    sieve on the increments of the input, not the input itself.

    Args:
        FeatureSieve (FeatureSieve): A feature sieve that will be used
            for the transformation.
    """

    def __init__(
        self,
        sieve: FeatureSieve,
    ) -> None:
        self._sieve = sieve

    @property
    def requires_fitting(self) -> bool:
        return self._sieve.requires_fitting

    def _nfeatures(self) -> int:
        return self._sieve.nfeatures()

    def _fit(self, X: np.ndarray) -> None:
        inc = _increments(X[:, np.newaxis, :], 1)[:, 0, :]
        self._sieve.fit(inc)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        inc = _increments(X[:, np.newaxis, :], 1)[:, 0, :]
        return self._sieve.transform(inc)

    def _copy(self) -> "INC":
        return INC(self._sieve.copy())

    def _summary(self) -> str:
        return f"INC>{self._sieve.summary()}"

    def __str__(self) -> str:
        return (f"INC({str(self._sieve)})")
