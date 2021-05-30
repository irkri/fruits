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

@numba.njit(parallel=True, fastmath=True)
def _max(X: np.ndarray):
    # accelerated function for fruits.features.MAX
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        if len(X[i]) == 0:
            continue
        maximum = X[i][0]
        for j in range(len(X[i])):
            if X[i][j] > maximum:
                maximum = X[i][j]
        result[i] = maximum
    return result

@numba.njit(parallel=True, fastmath=True)
def _min(X:np.ndarray):
    # accelerated function for fruits.features.MIN
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        if len(X[i]) == 0:
            continue
        minimum = X[i][0]
        for j in range(len(X[i])):
            if X[i][j] < minimum:
                minimum = X[i][j]
        result[i] = minimum
    return result

@numba.njit
def _coquantile(X: np.ndarray, q: float):
    # calculates R = <[11], ISS(X_i)> for every time series X_i in X
    # calculates the q-quantiles in R for every X_i
    # and an index in R of the last element less than or equal
    # to this q-quantile for every X_i
    # the (rounded) mean of those quantiles will be returned
    iss = _fast_ISS(np.expand_dims(X, 1),
                    np.array([[[2]]]),
                    np.array([0]))[:, 0, :]
    coquantiles = np.zeros(X.shape[0])
    relative_max = iss[:, -1] * q
    for i in range(X.shape[0]):
        coquantiles[i] = np.sum((iss[i] <= relative_max[i]))

    return int(coquantiles.min())
