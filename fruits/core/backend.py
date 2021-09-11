import numba
import numpy as np

from fruits.core.wording import Word

def _slow_single_ISS(Z: np.ndarray,
                     word: Word,
                     alphas: np.ndarray,
                     extended: int) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones(Z.shape[1], dtype=np.float64)
    for k, ext_letter in enumerate(word):
        C = np.ones(Z.shape[1], dtype=np.float64)
        for l in range(len(ext_letter)):
            C = C * ext_letter[l](Z[:, :])
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        tmp = tmp * np.exp(np.arange(Z.shape[1]) * (alphas[k+1]-alphas[k]))
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
    for l in range(len(extended_letter)):
        if extended_letter[l] != 0:
            C = C * Z[l, :]**extended_letter[l]
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
        if not np.any(word[k]):
            continue
        C = _fast_extended_letter(Z, word[k])
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        tmp = tmp * np.exp(np.arange(Z.shape[1]) * (alphas[k+1]-alphas[k]))
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
    for i in numba.prange(Z.shape[0]):
        result[i] = _fast_single_ISS(Z[i, :, :], word, alphas, extended)
    return result