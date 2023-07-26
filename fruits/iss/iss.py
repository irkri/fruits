from enum import Enum, auto
from typing import Generator, Literal, Optional, Sequence

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


class CosWISS(ISS):

    def __init__(
        self,
        freq: float,
        length: Literal[2, 3] = 2,
        squared: bool = False,
    ) -> None:
        self._freq = freq
        self._length: Literal[2, 3] = length
        self._squared = squared

    def n_iterated_sums(self) -> int:
        """Returns the number of iterated sums the object would return
        with a :meth:`~CosWISS.transform`` call.
        """
        return 1

    def _transform(self, X: np.ndarray) -> np.ndarray:
        result = self.batch_transform(X, batch_size=1)
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
            batch_size (int, optional): Option not available. Is only
                implemented for compatibility with default :class:`ISS`.
        """
        if batch_size > 1:
            raise ValueError("batch_size > 1 not supported for CosWISS")
        steps = np.arange(X.shape[2]) / (self._freq*(X.shape[2] - 1))
        sin_w = np.sin(np.pi * steps)[np.newaxis, np.newaxis, :]
        cos_w = np.cos(np.pi * steps)[np.newaxis, np.newaxis, :]
        if self._length == 2 and not self._squared:
            iss = np.cumsum(
                (X*sin_w)[:, :, 1:] * np.cumsum(X*sin_w, axis=2)[:, :, :-1],
                axis=2
            )
            iss += np.cumsum(
                (X*cos_w)[:, :, 1:] * np.cumsum(X*cos_w, axis=2)[:, :, :-1],
                axis=2
            )
            yield np.pad(iss, ((0, 0), (0, 0), (1, 0))).swapaxes(0, 1)
        elif self._length == 3 and not self._squared:
            iss = np.cumsum((X*cos_w)[:,:,2:] * np.cumsum(
                (X*cos_w**2)[:,:,1:] * np.cumsum(X*cos_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum((X*sin_w)[:,:,2:] * np.cumsum(
                (X*sin_w*cos_w)[:,:,1:] * np.cumsum(X*cos_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum((X*cos_w)[:,:,2:] * np.cumsum(
                (X*sin_w*cos_w)[:,:,1:] * np.cumsum(X*sin_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum((X*sin_w)[:,:,2:] * np.cumsum(
                (X*sin_w**2)[:,:,1:] * np.cumsum(X*sin_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            yield np.pad(iss, ((0, 0), (0, 0), (2, 0))).swapaxes(0, 1)
        elif self._length == 2 and self._squared:
            iss = np.cumsum(
                (X*sin_w**2)[:,:,1:] * np.cumsum(X*sin_w**2, axis=2)[:,:,:-1],
                axis=2
            )
            iss += 2 * np.cumsum(
                (X*sin_w*cos_w)[:,:,1:]
                    * np.cumsum(X*sin_w*cos_w, axis=2)[:,:,:-1],
                axis=2
            )
            iss += np.cumsum(
                (X*cos_w**2)[:,:,1:] * np.cumsum(X*cos_w**2, axis=2)[:,:,:-1],
                axis=2
            )
            yield np.pad(iss, ((0, 0), (0, 0), (1, 0))).swapaxes(0, 1)
        elif self._length == 3 and self._squared:
            iss = np.cumsum(
                (X*cos_w**2)[:,:,2:] * np.cumsum(
                    (X*cos_w**4)[:,:,1:] * np.cumsum(
                        X*cos_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            ) #
            iss += 2 * np.cumsum(
                (X*cos_w**2)[:,:,2:] * np.cumsum(
                    (X*cos_w**3*sin_w)[:,:,1:] * np.cumsum(
                        X*cos_w*sin_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum(
                (X*cos_w**2)[:,:,2:] * np.cumsum(
                    (X*cos_w**2*sin_w**2)[:,:,1:] * np.cumsum(
                        X*sin_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += 2 * np.cumsum(
                (X*cos_w*sin_w)[:,:,2:] * np.cumsum(
                    (X*cos_w**3*sin_w)[:,:,1:] * np.cumsum(
                        X*cos_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += 4 * np.cumsum(
                (X*cos_w*sin_w)[:,:,2:] * np.cumsum(
                    (X*cos_w**2*sin_w**2)[:,:,1:] * np.cumsum(
                        X*cos_w*sin_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += 2 * np.cumsum(
                (X*sin_w**2)[:,:,2:] * np.cumsum(
                    (X*cos_w*sin_w**3)[:,:,1:] * np.cumsum(
                        X*cos_w*sin_w, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum(
                (X*sin_w**2)[:,:,2:] * np.cumsum(
                    (X*cos_w**2*sin_w**2)[:,:,1:] * np.cumsum(
                        X*cos_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += 2 * np.cumsum(
                (X*sin_w*cos_w)[:,:,2:] * np.cumsum(
                    (X*cos_w*sin_w**3)[:,:,1:] * np.cumsum(
                        X*sin_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            iss += np.cumsum(
                (X*sin_w**2)[:,:,2:] * np.cumsum(
                    (X*sin_w**4)[:,:,1:] * np.cumsum(
                        X*sin_w**2, axis=2)[:,:,:-1],
                axis=2)[:,:,:-1], axis=2
            )
            yield np.pad(iss, ((0, 0), (0, 0), (2, 0))).swapaxes(0, 1)

    def _copy(self) -> "CosWISS":
        return CosWISS(
            freq=self._freq,
            length=self._length,
            squared=self._squared,
        )
