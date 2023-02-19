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
                result = self._iterated_sum_fast(
                    Z,
                    np.array(list(word)),
                    weighting.get_fast_args() if weighting is not None else (
                        np.zeros((len(word)+1, ), dtype=np.float32)
                    ),
                    extended,
                )
                return result
            except NotImplementedError:
                word = replace_letters(word, ("DIM" for _ in range(1)))
        result = np.zeros((Z.shape[0], extended, Z.shape[2]))
        for i in range(Z.shape[0]):
            result[i] = self._iterated_sum(Z[i], word, extended, weighting)
        return result

    @staticmethod
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        raise NotImplementedError("No fast way of calculating iterated sums")

    def _iterated_sum(
        self,
        Z: np.ndarray,
        word: Word,
        extended: int,
        weighting: Optional[Weighting] = None,
    ) -> np.ndarray:
        result = np.zeros((extended, Z.shape[1]), dtype=np.float64)
        tmp = self._identity(Z)
        for k, ext_letter in enumerate(word):
            C = self._identity(Z)
            for letter in ext_letter:
                C = self._operation(C, letter(Z))
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
            tmp[k:] = self._operation(tmp[k:], C[k:])
            if weighting is not None:
                tmp = self._operation(tmp, weighting.weights(Z.shape[1], k))
            tmp[k:] = self._cumulative_operation(tmp[k:])
            if len(word)-k <= extended:
                # save result
                result[extended-(len(word)-k), :] = tmp.copy()
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
        "float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
        fastmath=True,
        cache=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.ones((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for dim, el in enumerate(ext_letter):
                if el > 0:
                    for _ in range(el):
                        C = C * Z[:, dim, :]
                elif el < 0:
                    for _ in range(-el):
                        C = C / Z[:, dim, :]
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[:, 0] = 0
            tmp[:, k:] = tmp[:, k:] * C[:, k:]
            if alphas[k+1] != alphas[k] or alphas[k] != 0:
                tmp = tmp * np.exp(np.arange(Z.shape[2])
                                   * (alphas[k+1]-alphas[k])
                                   + alphas[k])
            for i in range(k+1, tmp.shape[1]):
                tmp[:, i] = tmp[:, i-1] + tmp[:, i]
            if len(word)-k <= extended:
                # save result
                result[:, extended-(len(word)-k), :] = tmp.copy()
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
        "float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for dim, el in enumerate(ext_letter):
                if el != 0:
                    C = C + el * Z[:, dim, :]
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[:, 0] = 0
            tmp[:, k:] = tmp[:, k:] + C[:, k:]
            # if alphas[k+1] != alphas[k] or alphas[k] != 0:
            #     tmp = tmp * np.exp(np.arange(Z.shape[1])
            #                     * (alphas[k+1]-alphas[k])
            #                     + alphas[k])
            for i in numba.prange(tmp.shape[0]):
                for j in range(k+1, tmp.shape[1]):
                    tmp[i, j] = min(tmp[i, j-1], tmp[i, j])
            if len(word)-k <= extended:
                # save result
                result[:, extended-(len(word)-k), :] = tmp.copy()
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
        "float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.zeros((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for dim, el in enumerate(ext_letter):
                if el != 0:
                    C = C + el * Z[:, dim, :]
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[:, 0] = 0
            tmp[:, k:] = tmp[:, k:] + C[:, k:]
            # if alphas[k+1] != alphas[k] or alphas[k] != 0:
            #     tmp = tmp * np.exp(np.arange(Z.shape[1])
            #                     * (alphas[k+1]-alphas[k])
            #                     + alphas[k])
            for i in numba.prange(tmp.shape[0]):
                for j in range(k+1, tmp.shape[1]):
                    tmp[i, j] = max(tmp[i, j-1], tmp[i, j])
            if len(word)-k <= extended:
                # save result
                result[:, extended-(len(word)-k), :] = tmp.copy()
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
        "float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _iterated_sum_fast(
        Z: np.ndarray,
        word: np.ndarray,
        alphas: np.ndarray,
        extended: int,
    ) -> np.ndarray:
        result = np.ones((Z.shape[0], extended, Z.shape[2]), dtype=np.float64)
        tmp = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
        for k, ext_letter in enumerate(word):
            if not np.any(ext_letter):
                continue
            C = np.ones((Z.shape[0], Z.shape[2]), dtype=np.float64)
            for dim, el in enumerate(ext_letter):
                if el > 0:
                    for _ in range(el):
                        C = C * Z[:, dim, :]
                elif el < 0:
                    for _ in range(-el):
                        C = C / Z[:, dim, :]
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[:, 0] = 0
            tmp[:, k:] = tmp[:, k:] + C[:, k:]
            # if alphas[k+1] != alphas[k] or alphas[k] != 0:
            #     tmp = tmp * np.exp(np.arange(Z.shape[1])
            #                     * (alphas[k+1]-alphas[k])
            #                     + alphas[k])
            for i in numba.prange(tmp.shape[0]):
                for j in range(k+1, tmp.shape[1]):
                    tmp[i, j] = max(tmp[i, j-1], tmp[i, j])
            if len(word)-k <= extended:
                # save result
                result[:, extended-(len(word)-k), :] = tmp.copy()
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
