from enum import Enum, auto
from typing import Generator, Optional, Sequence

import numpy as np

from ..seed import Seed
from .cache import CachePlan
from .semiring import ISSSemiRing, SumTimes
from .words.word import Word


class ISSMode(Enum):
    """Enum for different modes in an iterated sums calculation."""

    SINGLE = auto()
    EXTENDED = auto()


def _calculate_ISS(
    X: np.ndarray,
    words: Sequence[Word],
    batch_size: int,
    semiring: ISSSemiRing,
    cache_plan: Optional[CachePlan] = None,
) -> Generator[np.ndarray, None, None]:
    i = 0
    while i < len(words):
        if i + batch_size > len(words):
            batch_size = len(words) - i

        n_itsum = batch_size
        if cache_plan is not None:
            n_itsum = cache_plan.n_iterated_sums(range(i, i+batch_size))
        results = np.zeros((X.shape[0], n_itsum, X.shape[2]))

        index = 0
        for word in words[i:i+batch_size]:
            n_itsum_word = 1
            if cache_plan is not None:
                n_itsum_word = cache_plan.unique_el_depth(i)
            results[:, index:index+n_itsum_word, :] = semiring.iterated_sums(
                X,
                word,
                np.array([0.0] + word.alpha + [0.0], dtype=np.float32),
                n_itsum_word,
            )
            index += n_itsum_word
            i += 1

        yield results


class ISS(Seed):
    """Seed responsible for the calculation of iterated sums. Add this
    object with a number of words to a fruit.

    Args:
        words (Word or list of Words): Words to calculate the ISS for.
        mode (ISSMode, optional): Mode of the used calculator. Has to be
            either "single" or "extended".

            - SINGLE:
                Calculates one iterated sum for each given word.
            - EXTENDED:
                For each given word, the iterated sum for
                each sequential combination of extended letters
                in that word will be calculated. So for a simple
                word like ``[21][121][1]`` the calculator
                returns the iterated sums for ``[21]``,
                ``[21][121]`` and ``[21][121][1]``.
        semiring (ISSSemiring): The semi-ring in which the iterated sums
            are calculated. Defaults to :class:`SumTimes`.
    """

    def __init__(
        self,
        words: Sequence[Word],
        /, *,
        mode: ISSMode = ISSMode.SINGLE,
        semiring: Optional[ISSSemiRing] = None,
    ) -> None:
        self.words = words
        self.mode = mode
        self.semiring = semiring if semiring is not None else SumTimes()
        self._cache_plan = CachePlan(
            self.words if mode == ISSMode.EXTENDED else []
        )

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        result = _calculate_ISS(
            X,
            self.words,
            cache_plan=(
                self._cache_plan if self.mode == ISSMode.EXTENDED else None
            ),
            semiring=self.semiring,
            batch_size=len(self.words),

        )
        return next(iter(result))

    def n_iterated_sums(self) -> int:
        """Returns the number of iterated sums the object would return
        with a :meth:`~ISS.transform`` call.
        """
        if self.mode == ISSMode.EXTENDED:
            return self._cache_plan.n_iterated_sums()
        return len(self.words)

    def batch_transform(
        self,
        X: np.ndarray,
        batch_size: int = 1,
    ) -> Generator[np.ndarray, None, None]:
        """Yields a number of iterated sums one after another determined
        by the supplied ``batch_size``.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
            batch_size (int, optional): Number of words for which the
                iterated sums are calculated and returned at once. This
                doesn't have to be the same number as the number of
                words given. Default is that the iterated sums for each
                single word are returned one after another.
        """
        if batch_size > len(self.words):
            raise ValueError("batch_size too large, has to be < len(words)")
        yield from _calculate_ISS(
            X,
            self.words,
            cache_plan=(
                self._cache_plan if self.mode == ISSMode.EXTENDED else None
            ),
            semiring=self.semiring,
            batch_size=batch_size,
        )

    def _copy(self) -> "ISS":
        return ISS(self.words, mode=self.mode)

    def __str__(self) -> str:
        return (
            f"ISS(words={len(self.words)}, "
            f"mode={str(self.mode).split('.')[-1]})"
        )
