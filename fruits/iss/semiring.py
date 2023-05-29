from abc import ABC, abstractmethod
from typing import Optional

import numba
import numpy as np

from .weighting import Weighting
from .words.creation import replace_letters
from .words.word import SimpleWord, Word


class Semiring(ABC):

    def iterated_sums(
        self,
        Z: np.ndarray,
        word: Word,
        extended: int,
        weighting: Optional[Weighting] = None,
    ) -> np.ndarray:
        if isinstance(word, SimpleWord):
            try:
                if weighting is not None:
                    scalars, lookup = weighting.get_fast_args(
                        Z.shape[0], Z.shape[2]
                    )
                    if scalars is None:
                        scalars = np.ones((len(word)-1, ), dtype=np.float32)
                else:
                    scalars = np.zeros((len(word)-1, ), dtype=np.float32)
                    lookup = np.zeros((Z.shape[0], Z.shape[2]))
                result = self.iterated_sum_fast(
                    Z,
                    np.array(list(word), dtype=np.int32),
                    scalars,
                    lookup,
                    extended,
                )
                return result
            except NotImplementedError:
                word = replace_letters(word, ("DIM" for _ in range(1)))
        result = self._iterated_sum(Z, word, extended)
        return result

    def iterated_sum_fast(
        self,
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        raise NotImplementedError("No fast way of calculating iterated sums")

    def _iterated_sum(
        self,
        Z: np.ndarray,
        word: Word,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        for i in range(Z.shape[0]):
            tmp = self._identity(Z[i])
            for k, ext_letter in enumerate(word):
                C = self._identity(Z[i])
                for letter in ext_letter:
                    C = self._operation(C, letter(Z[i]))
                if k > 0:
                    tmp = np.roll(tmp, 1)
                    tmp[0] = 0
                tmp[k:] = self._operation(tmp[k:], C[k:])
                tmp[k:] = self._cumulative_operation(tmp[k:])
                if len(word)-k <= extended:
                    # save result
                    result[i, extended-(len(word)-k), :] = tmp.copy()
        return result

    @staticmethod
    @abstractmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        ...


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f8[:], i8)",
    fastmath=True,
    cache=True,
)
def _reals_single_iterated_sum_fast(
    Z: np.ndarray,
    word: np.ndarray,
    scalar: np.ndarray,
    weights: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.zeros((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones((Z.shape[1], ), dtype=np.float64)
    for k, extended_letter in enumerate(word):
        if not np.any(extended_letter):
            continue
        C = np.ones((Z.shape[1], ), dtype=np.float64)
        for letter, occurence in enumerate(extended_letter):
            if occurence > 0:
                for _ in range(occurence):
                    C = C * Z[letter, :]
            elif occurence < 0:
                for _ in range(-occurence):
                    C = C / Z[letter, :]
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp[k:] = tmp[k:] * C[k:]
        if k > 0:
            tmp = tmp * np.exp(- weights * scalar[k-1])
        if len(word) - k <= extended:
            result[extended-(len(word)-k), k:] = np.cumsum(tmp[k:])
        if k < len(word) - 1:
            tmp = tmp * np.exp(weights * scalar[k])
            tmp[k:] = np.cumsum(tmp[k:])
    return result


class Reals(Semiring):
    """This is the standard semiring used as the default in all
    computations of iterated sums. It is the field of real numbers with
    default summation and multiplication.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], f8[:,:], i8)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        for j in numba.prange(Z.shape[0]):
            result[j] = _reals_single_iterated_sum_fast(
                Z[j, :, :],
                word,
                scalar,
                lookup[j],
                extended,
            )
        return result

    def iterated_sum_fast(
        self,
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        if np.max(lookup[:, -1]) > 50:
            indices = np.where(lookup[:, -1] > 50)
            lookup[indices[0]] = lookup[indices[0]] * np.expand_dims(
                50 / lookup[indices[0], -1], 1
            )
        return self._iterated_sum_fast(
            Z,
            word,
            scalar,
            lookup,
            extended,
        )

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.cumsum(Z)


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f8[:])",
    fastmath=True,
    cache=True,
)
def _arctic_argmax_single_iterated_sum_fast(
    Z: np.ndarray,
    word: np.ndarray,
    scalar: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros((2*word.shape[0], Z.shape[1]), dtype=np.float64)
    tmp = np.zeros((Z.shape[1], ), dtype=np.float64)
    for k, ext_letter in enumerate(word):
        if not np.any(ext_letter):
            continue
        C = np.zeros(Z.shape[1], dtype=np.float64)
        for dim, el in enumerate(ext_letter):
            C = C + el * Z[dim, :]
        tmp = tmp + C
        if k > 0:
            tmp = tmp - weights * scalar[k-1]
        result[2*k, 0] = tmp[0]
        for i in range(1, Z.shape[1]):
            if result[2*k, i-1] >= tmp[i]:
                result[2*k, i] = result[2*k, i-1]
                result[2*k+1, i] = result[2*k+1, i-1]
            else:
                result[2*k, i] = tmp[i]
                result[2*k+1, i] = i
        if k < len(word) - 1:
            tmp = tmp + weights * scalar[k]
            for i in range(1, Z.shape[1]):
                tmp[i] = max(tmp[i-1], tmp[i])
    # translate indices back to their actual position
    n = int(word.shape[0] + (word.shape[0] * (word.shape[0]+1) / 2))
    translated_results = np.zeros((n, Z.shape[1]), dtype=np.float64)
    for k in range(word.shape[0]-1, -1, -1):
        index = int(k + (k * (k+1) / 2))
        translated_results[index] = result[2*k]
        translated_results[index+k+1] = result[2*k+1]
        for s in range(k, 0, -1):
            c = int(translated_results[index+s+1, -1])+1
            translated_results[index+s, :c] = result[2*(s-1)+1, :c]
            translated_results[index+s, c:] = result[2*(s-1)+1, c-1]
    return translated_results


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f8[:], i8)",
    fastmath=True,
    cache=True,
)
def _arctic_single_iterated_sum_fast(
    Z: np.ndarray,
    word: np.ndarray,
    scalar: np.ndarray,
    weights: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.zeros((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.zeros((Z.shape[1], ), dtype=np.float64)
    for k, ext_letter in enumerate(word):
        if not np.any(ext_letter):
            continue
        C = np.zeros(Z.shape[1], dtype=np.float64)
        for dim, el in enumerate(ext_letter):
            C = C + el * Z[dim, :]
        tmp = tmp + C
        if k > 0:
            tmp = tmp - weights * scalar[k-1]
        if len(word) - k <= extended:
            result[extended-(len(word)-k), 0] = tmp[0]
            for i in range(1, Z.shape[1]):
                result[extended-(len(word)-k), i] = max(
                    result[extended-(len(word)-k), i-1], tmp[i]
                )
        if k < len(word) - 1:
            tmp = tmp + weights * scalar[k]
            for i in range(1, Z.shape[1]):
                tmp[i] = max(tmp[i-1], tmp[i])
    return result


class Arctic(Semiring):
    """The arctic semiring is defined over the real numbers including
    negative infinity. The additive operation is the maximum and the
    multiplicative operation is the standard addition with units
    negative infinity and zero respectively.

    Args:
        argmax (bool, optional): Whether to additionally return the
            positions of all involved maxima. This leads to twice as
            many iterated sums to be returned. Each second one contains
            indices of the maxima in the previous.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], f8[:,:], i8, b1)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
        argmax: bool,
    ) -> np.ndarray:
        if argmax:
            n = int(word.shape[0] + (word.shape[0] * (word.shape[0]+1) / 2))
            result = np.zeros((Z.shape[0], n, Z.shape[2]), dtype=np.float64)
        else:
            result = np.zeros(
                (Z.shape[0], extended, Z.shape[2]),
                dtype=np.float64,
            )
        for j in numba.prange(Z.shape[0]):
            if argmax:
                result[j, :, :] = _arctic_argmax_single_iterated_sum_fast(
                    Z[j, :, :],
                    word,
                    scalar,
                    lookup[j],
                )
            else:
                result[j, :, :] = _arctic_single_iterated_sum_fast(
                    Z[j, :, :],
                    word,
                    scalar,
                    lookup[j],
                    extended,
                )
        return result

    def __init__(self, argmax: bool = False) -> None:
        self._argmax = argmax

    def iterated_sum_fast(
        self,
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        return self._iterated_sum_fast(
            Z,
            word,
            scalar,
            lookup,
            extended,
            self._argmax
        )

    def _iterated_sum(
        self,
        Z: np.ndarray,
        word: Word,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        for i in range(Z.shape[0]):
            tmp = self._identity(Z[i])
            for k, ext_letter in enumerate(word):
                C = self._identity(Z[i])
                for letter in ext_letter:
                    C = self._operation(C, letter(Z[i]))
                tmp = self._operation(tmp, C)
                tmp = self._cumulative_operation(tmp)
                if len(word)-k <= extended:
                    # save result
                    result[i, extended-(len(word)-k), :] = tmp.copy()
        return result

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(Z)


@numba.njit(
    "f8[:,:](f8[:,:], i4[:,:], f4[:], f8[:], i8)",
    fastmath=True,
    cache=True,
)
def _bayesian_single_iterated_sum_fast(
    Z: np.ndarray,
    word: np.ndarray,
    scalar: np.ndarray,
    weights: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones((Z.shape[1], ), dtype=np.float64)
    for k, extended_letter in enumerate(word):
        if not np.any(extended_letter):
            continue
        C = np.ones((Z.shape[1], ), dtype=np.float64)
        for letter, occurence in enumerate(extended_letter):
            if occurence > 0:
                for _ in range(occurence):
                    C = C * Z[letter, :]
            elif occurence < 0:
                for _ in range(-occurence):
                    C = C / Z[letter, :]
        tmp = tmp * C
        if k > 0:
            tmp = tmp * np.exp(- weights * scalar[k-1])
        if len(word)-k <= extended:
            result[extended-(len(word)-k), 0] = tmp[0]
            for i in range(1, Z.shape[1]):
                result[extended-(len(word)-k), i] = max(
                    result[extended-(len(word)-k), i-1], tmp[i]
                )
        if k < len(word) - 1:
            tmp = tmp * np.exp(weights * scalar[k])
            for i in range(1, Z.shape[1]):
                tmp[i] = max(tmp[i-1], tmp[i])
    return result


class Bayesian(Semiring):
    """The bayesian semiring is defined over the real numbers in [0,1].
    The additive operation is the maximum and the multiplicative
    operation is the standard multiplication with units zero and one
    respectively.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], f8[:,:], i8)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        for j in numba.prange(Z.shape[0]):
            result[j] = _bayesian_single_iterated_sum_fast(
                Z[j, :, :],
                word,
                scalar,
                lookup[j],
                extended,
            )
        return result

    def iterated_sum_fast(
        self,
        Z: np.ndarray,
        word: np.ndarray,
        scalar: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        if np.max(lookup[:, -1]) > 50:
            indices = np.where(lookup[:, -1] > 50)
            lookup[indices[0]] = lookup[indices[0]] * np.expand_dims(
                50 / lookup[indices[0], -1], 1
            )
        return self._iterated_sum_fast(
            Z,
            word,
            scalar,
            lookup,
            extended,
        )

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(Z)
