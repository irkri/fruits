from abc import ABC, abstractmethod
from typing import Optional

import numba
import numpy as np

from .words.word import Word, SimpleWord
from .words.creation import replace_letters
from .weighting import Weighting


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
                else:
                    scalars = np.zeros((len(word)+1, ), dtype=np.float32)
                    lookup = np.ones((Z.shape[0], Z.shape[2]))
                    lookup *= np.arange(Z.shape[2], dtype=np.float32)
                result = self._iterated_sum_fast(
                    Z,
                    np.array(list(word)),
                    scalars,
                    lookup,
                    extended,
                )
                return result
            except NotImplementedError:
                word = replace_letters(word, ("DIM" for _ in range(1)))
        result = self._iterated_sum(Z, word, extended)
        return result

    @staticmethod
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
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


class Reals(Semiring):
    """This is the standard semiring used as the default in all
    computations of iterated sums. It is the field of real numbers with
    default summation and multiplication.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], int32[:,:], "
                       "float32[:], float64[:,:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.ones((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for j in numba.prange(Z.shape[0]):
            for k, extended_letter in enumerate(word):
                if not np.any(extended_letter):
                    continue
                C = np.ones((Z.shape[2], ), dtype=np.float64)
                for letter, occurence in enumerate(extended_letter):
                    if occurence > 0:
                        for _ in range(occurence):
                            C = C * Z[j, letter, :]
                    elif occurence < 0:
                        for _ in range(-occurence):
                            C = C / Z[j, letter, :]
                if k > 0:
                    tmp[j] = np.roll(tmp[j], 1)
                    tmp[j, 0] = 0
                tmp[j, k:] = tmp[j, k:] * C[k:]
                if alphas.size > 1 and alphas[k+1] != alphas[k]:
                    tmp[j] = tmp[j] * np.exp(
                        (lookup[j] - lookup[j, -1]) * (alphas[k+1] - alphas[k])
                    )
                elif alphas.size == 1:
                    if k == 0:
                        tmp[j] = tmp[j] * np.exp(lookup[j] - lookup[j, -1])
                    elif k == len(word) - 1:
                        tmp[j] = tmp[j] * np.exp(-(lookup[j] - lookup[j, -1]))
                tmp[j, k:] = np.cumsum(tmp[j, k:])
                if len(word)-k <= extended:
                    result[j, extended-(len(word)-k), :] = tmp[j, :]
        return result

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.cumsum(Z)


class Tropical(Semiring):
    """The tropical semiring is defined over the real numbers including
    positive infinity. The additive operation is the minimum and the
    multiplicative operation is the standard addition with units
    positive infinity and zero respectively.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], int32[:,:], "
                       "float32[:], float64[:,:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for j in numba.prange(Z.shape[0]):
                for dim, el in enumerate(ext_letter):
                    if el != 0:
                        C[j] = C[j] + el * Z[j, dim, :]
                if k > 0:
                    tmp[j] = np.roll(tmp[j], 1)
                    tmp[j, 0] = 0
                tmp[j, k:] = tmp[j, k:] + C[j, k:]
                if alphas.size > 1 and alphas[k+1] != alphas[k]:
                    tmp[j] = tmp[j] + (
                        (lookup[j] / lookup[j, -1]) * (alphas[k+1] - alphas[k])
                    )
                elif alphas.size == 1:
                    if k == 0:
                        tmp[j] = tmp[j] + (lookup[j] / lookup[j, -1])
                    elif k == len(word) - 1:
                        tmp[j] = tmp[j] - (lookup[j] / lookup[j, -1])
                for i in range(k+1, tmp.shape[1]):
                    tmp[j, i] = min(tmp[j, i-1], tmp[j, i])
                if len(word)-k <= extended:
                    result[j, extended-(len(word)-k), :] = tmp[j]
        return result

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.minimum.accumulate(Z)


class Arctic(Semiring):
    """The arctic semiring is defined over the real numbers including
    negative infinity. The additive operation is the maximum and the
    multiplicative operation is the standard addition with units
    negative infinity and zero respectively.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], int32[:,:], "
                       "float32[:], float64[:,:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for j in numba.prange(Z.shape[0]):
                for dim, el in enumerate(ext_letter):
                    if el != 0:
                        C[j] = C[j] + el * Z[j, dim, :]
                if k > 0:
                    tmp[j] = np.roll(tmp[j], 1)
                    tmp[j, 0] = 0
                tmp[j, k:] = tmp[j, k:] + C[j, k:]
                if alphas.size > 1 and alphas[k+1] != alphas[k]:
                    tmp[j] = tmp[j] + (
                        (lookup[j] / lookup[j, -1]) * (alphas[k+1] - alphas[k])
                    )
                elif alphas.size == 1:
                    if k == 0:
                        tmp[j] = tmp[j] + (lookup[j] / lookup[j, -1])
                    elif k == len(word) - 1:
                        tmp[j] = tmp[j] - (lookup[j] / lookup[j, -1])
                for i in range(k+1, tmp.shape[1]):
                    tmp[j, i] = max(tmp[j, i-1], tmp[j, i])
                if len(word)-k <= extended:
                    result[j, extended-(len(word)-k), :] = tmp[j]
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


class Bayesian(Semiring):
    """The bayesian semiring is defined over the real numbers in [0,1].
    The additive operation is the maximum and the multiplicative
    operation is the standard multiplication with units zero and one
    respectively.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], int32[:,:], "
                       "float32[:], float64[:,:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        lookup: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.ones((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for j in numba.prange(Z.shape[0]):
            for k, extended_letter in enumerate(word):
                if not np.any(extended_letter):
                    continue
                C = np.ones((Z.shape[2], ), dtype=np.float64)
                for letter, occurence in enumerate(extended_letter):
                    if occurence > 0:
                        for _ in range(occurence):
                            C = C * Z[j, letter, :]
                    elif occurence < 0:
                        for _ in range(-occurence):
                            C = C / Z[j, letter, :]
                if k > 0:
                    tmp[j] = np.roll(tmp[j], 1)
                    tmp[j, 0] = 0
                tmp[j, k:] = tmp[j, k:] * C[k:]
                if alphas.size > 1 and alphas[k+1] != alphas[k]:
                    tmp[j] = tmp[j] * np.exp(
                        (lookup[j] - lookup[j, -1]) * (alphas[k+1] - alphas[k])
                    )
                elif alphas.size == 1:
                    if k == 0:
                        tmp[j] = tmp[j] * np.exp(lookup[j] - lookup[j, -1])
                    elif k == len(word) - 1:
                        tmp[j] = tmp[j] * np.exp(-(lookup[j] - lookup[j, -1]))
                for i in range(k+1, tmp.shape[1]):
                    tmp[j, i] = max(tmp[j, i-1], tmp[j, i])
                if len(word)-k <= extended:
                    # save result
                    result[j, extended-(len(word)-k), :] = tmp[j]
        return result

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(Z)
