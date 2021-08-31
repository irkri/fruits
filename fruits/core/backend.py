import numba
import numpy as np

def _slow_ISS(Z: np.ndarray, word: list) -> np.ndarray:
    # calculates the iterated sums for Z and a given list of
    # general ComplexWords
    result = np.zeros((Z.shape[0], Z.shape[2]))
    for i in range(Z.shape[0]):
        result[i, :] = np.ones(Z.shape[2], dtype=np.float64)
        r = len(word)
        for k, el in enumerate(word):
            C = np.ones(Z.shape[2], dtype=np.float64)
            for l in range(len(el)):
                C = C * el[l](Z[i, :, :])
            result[i, :] = _fast_CS(result[i, :] * C,
                                    int(bool(r-(k+1))))
    return result

@numba.njit("float64[:](float64[:], int32)", cache=True)
def _fast_CS(Z: np.ndarray, r: int):
    Z = np.roll(np.cumsum(Z), r)
    Z[:r] = 0
    return Z

@numba.njit("float64[:](float64[:,:], int32[:,:])",
            fastmath=True, cache=True)
def _fast_single_ISS(Z: np.ndarray, word: np.ndarray) -> np.ndarray:
    result = np.ones(Z.shape[1], dtype=np.float64)
    r = len(word)
    for k in range(r):
        if not np.any(word[k]):
            continue
        C = np.ones(Z.shape[1], dtype=np.float64)
        for l in range(len(word[k])):
            if word[k][l] != 0:
                C = C * Z[l, :]**word[k][l]
        result = _fast_CS(result * C, int(bool(r-(k+1))))
    return result

@numba.njit("float64[:,:](float64[:,:,:], int32[:,:])",
            parallel=True, cache=True)
def _fast_ISS(Z: np.ndarray, word: np.ndarray) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        result[i, :] = _fast_single_ISS(Z[i, :, :], word)
    return result
