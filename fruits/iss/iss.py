from enum import Enum, auto
from typing import Generator, Literal, Optional, Sequence

import numba
import numpy as np

from ..cache import SharedSeedCache
from ..seed import Seed
from .cache import CachePlan
from .semiring import Arctic, Reals, Semiring
from .weighting import Weighting
from .words.word import SimpleWord, Word


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
        elif isinstance(semiring, Arctic) and semiring._argmax:
            raise NotImplementedError(
                "Arctic argmax is not implemented when using ISSMode.SINGLE"
            )
        if isinstance(semiring, Arctic) and semiring._argmax:
            n_itsum = sum(
                len(word) + int(len(word) * (len(word)+1) / 2)
                for word in words[i:i+batch_size]
            )
        results = np.zeros((n_itsum, X.shape[0], X.shape[2]))

        index = 0
        for word in words[i:i+batch_size]:
            n_itsum_word = 1 if cache_plan is None else (
                cache_plan.unique_el_depth(i)
            )
            if isinstance(semiring, Arctic) and semiring._argmax:
                n_itsum_word = len(word) + int(len(word) * (len(word)+1) / 2)
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
        words (Sequence of Words): Words to calculate the ISS for.
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
                return sum(
                    len(word) + int(len(word) * (len(word)+1) / 2)
                    for word in self.words
                )
            return self._cache_plan.n_iterated_sums()
        elif isinstance(self.semiring, Arctic) and self.semiring._argmax:
            raise NotImplementedError(
                "Arctic argmax is not implemented when using ISSMode.SINGLE"
            )
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


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f4, i4[:,:])",
    fastmath=True,
    cache=True,
)
def _coswiss_single(
    X: np.ndarray,
    word: np.ndarray,
    freq: float,
    weightings: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[1], ))
    sin_w = np.sin(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    cos_w = np.cos(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    for i in range(weightings.shape[0]):
        tmp = np.ones((X.shape[1], ), dtype=np.float64)
        for k, extended_letter in enumerate(word):
            C = np.ones((X.shape[1], ), dtype=np.float64)
            for letter, occurence in enumerate(extended_letter):
                if occurence > 0:
                    for _ in range(occurence):
                        C = C * X[letter, :]
                elif occurence < 0:
                    for _ in range(-occurence):
                        C = C / X[letter, :]
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
            tmp[k:] = np.cumsum(tmp[k:] * C[k:] * (
                sin_w[k:]**weightings[i, 2*k+1]*cos_w[k:]**weightings[i, 2*k+2]
            ))
        result += weightings[i, 0] * tmp
    return result


@numba.njit(
    "f8[:,:](f8[:,:,:], i4[:,:], f4, i4[:,:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _coswiss(
    X: np.ndarray,
    word: np.ndarray,
    freq: float,
    weightings: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[0], X.shape[2]))
    for i in numba.prange(X.shape[0]):
        result[i, :] = _coswiss_single(X[i, :, :], word, freq, weightings)
    return result


class CosWISS(ISS):
    """The Cosine Weighted ISS is a regular ISS over the
    :class:`~fruits.iss.semiring.Reals` semiring with squared
    cosine weighted summands, e.g.::

        cos(pi * |i-j| / (f*N))**2 * x_i * x_j

    Args:
        freqs (list of floats): Frequencies ``f`` to calculate the
            weighted ISS for.
        words (Sequence of Words): Words to calculate the ISS for.
            Currently only words of length 2 and 3 are supported.
        squared (bool, optional): If set to false, the cosine will not
            be squared. Defaults to true.
    """

    def __init__(
        self,
        freqs: Sequence[float],
        words: Sequence[Word],
        squared: bool = True,
    ) -> None:
        self._freqs = freqs
        for word in words:
            if not isinstance(word, SimpleWord):
                raise ValueError("CosWISS only implemented for simple words")
            if len(word) not in [2, 3]:
                raise ValueError(
                    f"CosWISS not implemented for word of length {len(word)}"
                )
        self._words = words
        self._squared = squared

    def n_iterated_sums(self) -> int:
        """Total number of iterated sums the current ISS configuration
        produces.
        """
        return len(self._freqs) * len(self._words)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        result = self.batch_transform(X, batch_size=self.n_iterated_sums())
        return next(iter(result))

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
            batch_size (int, optional): Number of iterated sums returned
                at once. Default is 1.
        """
        results = []
        c = 0
        for freq in self._freqs:
            for word in self._words:
                weightings = np.empty((1, 1))
                if len(word) == 2 and not self._squared:
                    weightings = np.array([
                        [1, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                    ], dtype=np.int32)
                elif len(word) == 3 and not self._squared:
                    weightings = np.array([
                        [1, 0, 1, 0, 2, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0],
                        [1, 1, 0, 1, 1, 0, 1],
                        [1, 1, 0, 2, 0, 1, 0],
                    ], dtype=np.int32)
                elif len(word) == 2 and self._squared:
                    weightings = np.array([
                        [1, 2, 0, 2, 0],
                        [2, 1, 1, 1, 1],
                        [1, 0, 2, 0, 2],
                    ], dtype=np.int32)
                elif len(word) == 3 and self._squared:
                    weightings = np.array([
                        [1, 0, 2, 0, 4, 0, 2],
                        [2, 1, 1, 1, 3, 0, 2],
                        [1, 2, 0, 2, 2, 0, 2],
                        [2, 0, 2, 1, 3, 1, 1],
                        [4, 1, 1, 2, 2, 1, 1],
                        [2, 1, 1, 3, 1, 2, 0],
                        [1, 0, 2, 2, 2, 2, 0],
                        [2, 2, 0, 3, 1, 1, 1],
                        [1, 2, 0, 4, 0, 2, 0],
                    ], dtype=np.int32)
                else:
                    raise RuntimeError("Unsupported CosWISS configuration")
                results.append(_coswiss(
                    X,
                    np.array(list(word), dtype=np.int32),
                    freq,
                    weightings,
                ))
                if len(results) == batch_size:
                    yield np.array(results, dtype=results[0].dtype)
                    results = []
        if len(results) != 0:
            yield np.array(results, dtype=results[0].dtype)

    def _copy(self) -> "CosWISS":
        return CosWISS(
            freqs=self._freqs,
            words=self._words,
            squared=self._squared,
        )
