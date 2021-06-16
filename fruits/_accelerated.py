import numba
import numpy as np

@numba.njit(parallel=True, fastmath=True)
def _fast_ISS(Z: np.ndarray, 
              words: np.ndarray) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], len(words), Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        for j in numba.prange(len(words)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            for k in range(len(words[j])):
                if not np.any(words[j][k]):
                    continue
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(words[j][k])):
                    if words[j][k][l] != 0:
                        C = C * Z[i, l, :]**words[j][k][l]
                result[i, j, :] = np.cumsum(result[i, j, :] * C)
    return result

@numba.njit(fastmath=True)
def _increments(X: np.ndarray):
    # accelerated function that calculates increments of every
    # time series in X, the first value is the first value of the
    # time series
    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result[i, j, 0] = X[i, j, 0]
            for k in range(1, X.shape[2]):
                result[i, j, k] = X[i, j, k] - X[i, j, k-1]
    return result

@numba.njit()
def _coquantile(X: np.ndarray, q: float):
    # calculates R = <[11], ISS(X)> and an index in R of the last
    # element less than or equal to q * max(R)
    # for q in (0, 1) the result is an integer in (0, len(X)]
    X_inc = _increments(np.expand_dims(np.expand_dims(X, axis=0), axis=1))
    iss = _fast_ISS(X_inc, np.array([[[2]]]))[0, 0, :]
    relative_max = iss[-1] * q
    return np.sum((iss <= relative_max))
