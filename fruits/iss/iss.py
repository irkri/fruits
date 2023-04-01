from enum import Enum, auto
from typing import Generator, Optional, Sequence

import numpy as np

from ..cache import SharedSeedCache
from ..seed import Seed
from .cache import CachePlan
from .semiring import Reals, Arctic, Semiring
from .weighting import Weighting
from .words.word import Word


class ISSMode(Enum):
    """Enum for different modes in an iterated sums calculation."""

    SINGLE = auto()
    EXTENDED = auto()


def _calculate_ISS(
    X: np.ndarray,
    words: Sequence[Word],
    batch_size: int,
    semiring: Semiring,
    weighting: Optional[Weighting] = None,
    cache_plan: Optional[CachePlan] = None,
) -> Generator[np.ndarray, None, None]:
    i = 0
    while i < len(words):
        if i + batch_size > len(words):
            batch_size = len(words) - i

        n_itsum = batch_size
        if cache_plan is not None:
            n_itsum = cache_plan.n_iterated_sums(range(i, i+batch_size))
        if isinstance(semiring, Arctic) and semiring._argmax:
            n_itsum *= 2
        results = np.zeros((n_itsum, X.shape[0], X.shape[2]))

        index = 0
        for word in words[i:i+batch_size]:
            n_itsum_word = 1 if cache_plan is None else (
                cache_plan.unique_el_depth(i)
            )
            if isinstance(semiring, Arctic) and semiring._argmax:
                n_itsum_word *= 2
            results[index:index+n_itsum_word, :, :] = np.swapaxes(
                semiring.iterated_sums(
                    X,
                    word,
                    1 if cache_plan is None else cache_plan.unique_el_depth(i),
                    weighting,
                ),
                0, 1,
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
        semiring (Semiring, optional): The semiring in which the
            iterated sums are calculated. Defaults to :class:`Reals`.
        weighting (Weighting, optional): A specific weighting used for
            the iterated sums that penalize terms which involve indices
            in the time series that are further apart.
    """

    def __init__(
        self,
        words: Sequence[Word],
        /, *,
        mode: ISSMode = ISSMode.SINGLE,
        semiring: Optional[Semiring] = None,
        weighting: Optional[Weighting] = None,
    ) -> None:
        self.words = words
        self.mode = mode
        self.semiring = semiring if semiring is not None else Reals()
        self._cache_plan = CachePlan(
            self.words if mode == ISSMode.EXTENDED else []
        )
        self.weighting = weighting

    @property
    def requires_fitting(self) -> bool:
        return False

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "_cache") and self.weighting is not None:
            self.weighting._cache = self._cache
        elif self.weighting is not None:
            self.weighting._cache = SharedSeedCache(X)
        result = _calculate_ISS(
            X,
            self.words,
            cache_plan=(
                self._cache_plan if self.mode == ISSMode.EXTENDED else None
            ),
            semiring=self.semiring,
            weighting=self.weighting,
            batch_size=len(self.words),
        )
        return next(iter(result))

    def n_iterated_sums(self) -> int:
        """Returns the number of iterated sums the object would return
        with a :meth:`~ISS.transform`` call.
        """
        if self.mode == ISSMode.EXTENDED:
            if isinstance(self.semiring, Arctic) and self.semiring._argmax:
                return self._cache_plan.n_iterated_sums() * 2
            return self._cache_plan.n_iterated_sums()
        if isinstance(self.semiring, Arctic) and self.semiring._argmax:
            return len(self.words) * 2
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
        if hasattr(self, "_cache") and self.weighting is not None:
            self.weighting._cache = self._cache
        elif self.weighting is not None:
            self.weighting._cache = SharedSeedCache(X)
        yield from _calculate_ISS(
            X,
            self.words,
            cache_plan=(
                self._cache_plan if self.mode == ISSMode.EXTENDED else None
            ),
            semiring=self.semiring,
            weighting=self.weighting,
            batch_size=batch_size,
        )

    def _copy(self) -> "ISS":
        return ISS(
            self.words,
            mode=self.mode,
            semiring=self.semiring,
            weighting=self.weighting,
        )
