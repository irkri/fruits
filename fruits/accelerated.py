import numba
import numpy as np

@numba.njit(parallel=True, fastmath=True)
def _fast_ISS(Z: np.ndarray, 
              iterators: np.ndarray,
              scales: np.ndarray) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], len(iterators), Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        for j in numba.prange(len(iterators)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            for k in range(len(iterators[j])):
                if not np.any(iterators[j][k]):
                    continue
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(iterators[j][k])):
                    if iterators[j][k][l] != 0:
                        C = C * Z[i, l, :]**iterators[j][k][l]
                result[i, j, :] = np.cumsum(result[i, j, :] * C)
                result[i, j, :] /= Z.shape[2]**scales[j]
    return result

@numba.njit(parallel=True, fastmath=True)
def _fast_ppv(X: np.ndarray, ref_value: float) -> np.ndarray:
    # accelerated function for fruits.features.PPV
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        c = 0
        for j in range(len(X[i])):
            if X[i][j] >= ref_value:
                c += 1
        if len(X[i]) == 0:
            result[i] = 0
        else:
            result[i] = c / len(X[i])
    return result

@numba.njit(parallel=True)
def _max(X: np.ndarray, cut: float):
    # accelerated function for fruits.features.MAX
    result = np.zeros(X.shape[0])
    for i in numba.prange(X.shape[0]):
        this_cut = cut
        if 0 < this_cut < 1:
            this_cut = _coquantile(X[i, :], this_cut)
        elif this_cut < 0:
            this_cut = X.shape[1]
        elif this_cut > X.shape[1]:
            raise IndexError("Cutting index out of range")
        maximum = X[i][0]
        for j in range(this_cut):
            if X[i][j] > maximum:
                maximum = X[i][j]
        result[i] = maximum
    return result

@numba.njit(parallel=True)
def _min(X: np.ndarray, cut: float):
    # accelerated function for fruits.features.MIN
    result = np.zeros(X.shape[0])
    for i in numba.prange(X.shape[0]):
        this_cut = cut
        if 0 < this_cut < 1:
            this_cut = _coquantile(X[i, :], this_cut)
        elif this_cut < 0:
            this_cut = X.shape[1]
        elif this_cut > X.shape[1]:
            raise IndexError("Cutting index out of range")
        minimum = X[i][0]
        for j in range(this_cut):
            if X[i][j] < minimum:
                minimum = X[i][j]
        result[i] = minimum
    return result

@numba.njit
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

@numba.njit
def _coquantile(X: np.ndarray, q: float):
    # calculates R = <[11], ISS(X)> and an index in R of the last
    # element less than or equal to q * max(R)
    # for q in (0, 1) the result is an integer in (0, len(X)]
    X_inc = _increments(np.expand_dims(np.expand_dims(X, axis=0), axis=1))
    iss = _fast_ISS(X_inc, np.array([[[2]]]), np.array([0]))[0, 0, :]
    relative_max = iss[-1] * q
    return np.sum((iss <= relative_max))
