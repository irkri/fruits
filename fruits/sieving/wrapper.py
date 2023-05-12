__all__ = ["INC"]

import numpy as np

from ..cache import _increments
from .abstract import FeatureSieve


class INC(FeatureSieve):
    """Wrapper for another feature sieve. INC will evaluate the given
    sieve on the increments of the input, not the input itself.

    Args:
        sieve (FeatureSieve): A feature sieve that will be used
            for the transformation.
        depth (int, optional): Depth of the increments to compute before
            applying the given sieve. Same as doing
            ``INC(INC(...INC(sieve)))`` ``depth`` times. Defaults to 1.
        shift (int or float, optional): If an integer is given, the time
            series is shifted this number of indices bevor subtracting
            it from the unshifted version. So ``shift=1`` are the
            standard increments. Defaults to 1.
    """

    def __init__(
        self,
        sieve: FeatureSieve,
        depth: int = 1,
        shift: int = 1,
    ) -> None:
        self._sieve = sieve
        self._shift = shift
        self._depth = depth

    @property
    def requires_fitting(self) -> bool:
        return self._sieve.requires_fitting

    def _nfeatures(self) -> int:
        return self._sieve.nfeatures()

    def _fit(self, X: np.ndarray) -> None:
        inc = X[:, np.newaxis, :]
        for _ in range(self._depth):
            inc = _increments(X[:, np.newaxis, :], self._shift)
        self._sieve.fit(inc[:, 0, :])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        inc = X[:, np.newaxis, :]
        for _ in range(self._depth):
            inc = _increments(X[:, np.newaxis, :], self._shift)
        return self._sieve.transform(inc[:, 0, :])

    def _copy(self) -> "INC":
        return INC(self._sieve.copy(), depth=self._depth, shift=self._shift)

    def _summary(self) -> str:
        return f"INC>{self._sieve.summary()}"

    def __str__(self) -> str:
        return f"INC({str(self._sieve)}, {self._depth}, {self._shift})"
