import numba
import numpy as np

from fruits.words.word import Word


def _slow_single_ISS(Z: np.ndarray,
                     word: Word,
                     alphas: np.ndarray,
                     extended: int) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones(Z.shape[1], dtype=np.float64)
    for k, ext_letter in enumerate(word):
        C = np.ones(Z.shape[1], dtype=np.float64)
        for el in range(len(ext_letter)):
            C = C * ext_letter[el](Z[:, :])
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        if alphas[k+1] != alphas[k] or alphas[k] != 0:
            tmp = tmp * np.exp(np.arange(Z.shape[1])
                               * (alphas[k+1]-alphas[k])
                               + alphas[k])
        tmp = _fast_CS(tmp)
        if len(word)-k <= extended:
            # save result
            result[extended-(len(word)-k), :] = tmp.copy()
    return result


def _slow_ISS(Z: np.ndarray,
              word: Word,
              alphas: np.ndarray,
              extended: int) -> np.ndarray:
    # calculates iterated sums of Z for a list of general words
    result = np.ones((Z.shape[0], extended, Z.shape[2]))
    for i in range(Z.shape[0]):
        result[i] = _slow_single_ISS(Z[i], word, alphas, extended)
    return result


@numba.njit("float64[:](float64[:])", cache=True)
def _fast_CS(Z: np.ndarray) -> np.ndarray:
    return np.cumsum(Z)


@numba.njit("float64[:](float64[:, :], int32[:])", cache=True)
def _fast_extended_letter(Z: np.ndarray,
                          extended_letter: np.ndarray) -> np.ndarray:
    C = np.ones(Z.shape[1], dtype=np.float64)
    for el in range(len(extended_letter)):
        if extended_letter[el] != 0:
            C = C * Z[el, :]**extended_letter[el]
    return C


@numba.njit("float64[:,:](float64[:,:], int32[:,:], float32[:], int32)",
            fastmath=True, cache=True)
def _fast_single_ISS(Z: np.ndarray,
                     word: np.ndarray,
                     alphas: np.ndarray,
                     extended: int) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones(Z.shape[1], dtype=np.float64)
    for k, ext_letter in enumerate(word):
        if not np.any(ext_letter):
            continue
        C = _fast_extended_letter(Z, ext_letter)
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        if alphas[k+1] != alphas[k] or alphas[k] != 0:
            tmp = tmp * np.exp(np.arange(Z.shape[1])
                               * (alphas[k+1]-alphas[k])
                               + alphas[k])
        tmp = _fast_CS(tmp)
        if len(word)-k <= extended:
            # save result
            result[extended-(len(word)-k), :] = tmp.copy()
    return result


@numba.njit("float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
            parallel=True, cache=True)
def _fast_ISS(Z: np.ndarray,
              word: np.ndarray,
              alphas: np.ndarray,
              extended: int) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], extended, Z.shape[2]))
    for i in numba.prange(Z.shape[0]):  # pylint: disable=not-an-iterable
        result[i] = _fast_single_ISS(Z[i, :, :], word, alphas, extended)
    return result
