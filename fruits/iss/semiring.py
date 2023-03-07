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
    "float64[:,:]("
        "float64[:,:], int32[:,:], float32[:], float64[:], int64)",
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
        if k > 0 and len(word) > 1:
            tmp = tmp * np.exp(- weights * scalar[k-1])
        if len(word) - k <= extended:
            result[extended-(len(word)-k), k:] = np.cumsum(tmp[k:])
        if k < len(word) - 1 and len(word) > 1:
            tmp = tmp * np.exp(weights * scalar[k])
            tmp[k:] = np.cumsum(tmp[k:])
    return result


class Reals(Semiring):
    """This is the standard semiring used as the default in all
    computations of iterated sums. It is the field of real numbers with
    default summation and multiplication.
    """

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
    @numba.njit(
        "float64[:,:,:]("
            "float64[:,:,:], int32[:,:], float32[:], float64[:,:], int64)",
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
    "float64[:,:]("
        "float64[:,:], int32[:,:], float32[:], float64[:], int64)",
    fastmath=True,
    cache=True,
)
def _tropical_single_iterated_sum_fast(
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
            if el != 0:
                C = C + el * Z[dim, :]
        tmp = tmp + C
        if k > 0 and len(word) > 1:
            tmp = tmp - weights * scalar[k-1]
        if len(word) - k <= extended:
            result[extended-(len(word)-k), 0] = tmp[0]
            for i in range(1, Z.shape[1]):
                result[extended-(len(word)-k), i] = min(
                    result[extended-(len(word)-k), i-1], tmp[i]
                )
        if k < len(word) - 1 and len(word) > 1:
            tmp = tmp + weights * scalar[k]
            for i in range(1, Z.shape[1]):
                tmp[i] = min(tmp[i-1], tmp[i])
    return result


class Tropical(Semiring):
    """The tropical semiring is defined over the real numbers including
    positive infinity. The additive operation is the minimum and the
    multiplicative operation is the standard addition with units
    positive infinity and zero respectively.
    """

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
        )

    @staticmethod
    @numba.njit(
        "float64[:,:,:]("
            "float64[:,:,:], int32[:,:], float32[:], float64[:,:], int64)",
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
            result[j] = _tropical_single_iterated_sum_fast(
                Z[j, :, :],
                word,
                scalar,
                lookup[j],
                extended,
            )
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


@numba.njit(
    "float64[:,:]("
        "float64[:,:], int32[:,:], float32[:], float64[:], int64)",
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
            if el != 0:
                C = C + el * Z[dim, :]
        tmp = tmp + C
        if k > 0 and len(word) > 1:
            tmp = tmp - weights * scalar[k-1]
        if len(word) - k <= extended:
            result[extended-(len(word)-k), 0] = tmp[0]
            for i in range(1, Z.shape[1]):
                result[extended-(len(word)-k), i] = max(
                    result[extended-(len(word)-k), i-1], tmp[i]
                )
        if k < len(word) - 1 and len(word) > 1:
            tmp = tmp + weights * scalar[k]
            for i in range(1, Z.shape[1]):
                tmp[i] = max(tmp[i-1], tmp[i])
    return result


class Arctic(Semiring):
    """The arctic semiring is defined over the real numbers including
    negative infinity. The additive operation is the maximum and the
    multiplicative operation is the standard addition with units
    negative infinity and zero respectively.
    """

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
        )

    @staticmethod
    @numba.njit(
        "float64[:,:,:]("
            "float64[:,:,:], int32[:,:], float32[:], float64[:,:], int64)",
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
            result[j] = _arctic_single_iterated_sum_fast(
                Z[j, :, :],
                word,
                scalar,
                lookup[j],
                extended,
            )
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
    "float64[:,:]("
        "float64[:,:], int32[:,:], float32[:], float64[:], int64)",
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
        if k > 0 and len(word) > 1:
            tmp = tmp * np.exp(- weights * scalar[k-1])
        if len(word)-k <= extended:
            result[extended-(len(word)-k), 0] = tmp[0]
            for i in range(1, Z.shape[1]):
                result[extended-(len(word)-k), i] = max(
                    result[extended-(len(word)-k), i-1], tmp[i]
                )
        if k < len(word) - 1 and len(word) > 1:
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
    @numba.njit(
        "float64[:,:,:]("
            "float64[:,:,:], int32[:,:], float32[:], float64[:,:], int64)",
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

    @staticmethod
    def _identity(X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[1], dtype=np.float64)

    @staticmethod
    def _operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    @staticmethod
    def _cumulative_operation(Z: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(Z)
