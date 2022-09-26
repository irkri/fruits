import numba
import numpy as np


@numba.njit("float64[:,:,:](float64[:,:,:])", cache=True)
def _increments(X: np.ndarray) -> np.ndarray:
    # calculates the increments of each time series in X
    result = np.zeros(X.shape)
    result[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]  # type: ignore
    return result


@numba.njit("int64[:](float64[:,:,:], float32)", parallel=False, cache=True)
def _coquantile(X: np.ndarray, q: float) -> np.ndarray:
    # calculates the coquantiles of each time series in X for given q
    Y = _increments(X)[:, 0, :]
    results = np.zeros(Y.shape[0], dtype=np.int64)
    for i in numba.prange(Y.shape[0]):  # pylint: disable=not-an-iterable
        Y[i, :] = np.cumsum(Y[i, :]**2)
        results[i] = np.sum(Y[i, :] <= q * Y[i, -1])
    return results
