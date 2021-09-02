import numba
import numpy as np

def _slow_ISS(Z: np.ndarray, word: list, alphas: np.ndarray) -> np.ndarray:
    # calculates the iterated sums for Z and a given list of
    # general Words
    result = np.ones((Z.shape[0], Z.shape[2]))
    for i in range(Z.shape[0]):
        for k, el in enumerate(word):
            C = np.ones(Z.shape[2], dtype=np.float64)
            for l in range(len(el)):
                C = C * el[l](Z[i, :, :])
            tmp = result[i, :] * C
            tmp = tmp * np.exp(np.arange(Z.shape[2]) * (alphas[k+1]-alphas[k]))
            result[i, :] = _fast_CS(tmp)
            if k < len(word)-1:
                result[i, :] = np.roll(result[i, :], 1)
                result[i, 0] = 0
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

@numba.njit("float64[:,:](float64[:,:], int32[:,:], float32[:], boolean)",
            fastmath=True, cache=True)
def _fast_single_ISS(Z: np.ndarray,
                     word: np.ndarray,
                     alphas: np.ndarray,
                     extended: bool) -> np.ndarray:
    if extended:
        result = np.ones((len(word), Z.shape[1]), dtype=np.float64)
    else:
        result = np.ones((1, Z.shape[1]), dtype=np.float64)
    for k in range(len(word)):
        if not np.any(word[k]):
            continue
        C = _fast_extended_letter(Z, word[k])
        if k == 0 or not extended:
            tmp = result[0] * C
        else:
            tmp = np.roll(result[k-1], 1)
            tmp[0] = 0
            tmp = tmp * C
        tmp = tmp * np.exp(np.arange(Z.shape[1]) * (alphas[k+1]-alphas[k]))
        if extended:
            result[k] = _fast_CS(tmp)
        else:
            result[0] = _fast_CS(tmp)
            if k < len(word)-1:
                result[0] = np.roll(result[0], 1)
                result[0, 0] = 0
    return result

@numba.njit("float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], boolean)",
            parallel=True, cache=True)
def _fast_ISS(Z: np.ndarray,
              word: np.ndarray,
              alphas: np.ndarray,
              extended: bool) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    if extended:
        result = np.zeros((Z.shape[0], len(word), Z.shape[2]))
    else:
        result = np.zeros((Z.shape[0], 1, Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        result[i] = _fast_single_ISS(Z[i, :, :], word, alphas, extended)
    return result
