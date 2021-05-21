import numpy as np
import numba
from fruits.iterators import SummationIterator, SimpleWord

def ISS(Z: np.ndarray, iterators: list) -> np.ndarray:
    """Top level function that takes in a number of time series and a
    list of SummationIterators and decides which function to use at each
    iterator to get the best performance possible.

    For each given time series Z, this function returns the iteratively 
    calulcated cummulative sums of the input data, which will be 
    stepwise transformed using the specified SummationIterator[s].
    :param Z: three dimensional array containing multidimensional 
    time series data
    :type Z: numpy.ndarray
    :param iterators: list of objects of type SummationIterator
    :type iterators: list
    :returns: numpy array of shape (Z.shape[0], len(iterators), 
    Z.shape[2])
    :rtype: {numpy.ndarray}
    """
    if isinstance(iterators, SummationIterator):
        iterators = [iterators]

    # divide iterators into seperate lists for SimpleWords and other 
    # SummationIterators
    fast_iterators = []
    fast_iterators_index = []
    slow_iterators = []
    slow_iterators_index = []
    for i, iterator in enumerate(iterators):
        if isinstance(iterator, SimpleWord):
            fast_iterators.append(iterator)
            fast_iterators_index.append(i)
        elif isinstance(iterator, SummationIterator):
            slow_iterators.append(iterator)
            slow_iterators_index.append(i)
        else:
            raise TypeError("Iterators should be of type SummationIterator")

    # get solution for the fast iterators
    # we have to transform the SimpleWords first such that each word has 
    # the same length and monomial length
    if fast_iterators:
        max_dim = max(x.max_dim for x in fast_iterators)
        fast_iterators_raw = [list(x.monomials()) for x in fast_iterators]
        max_word_length = max(len(x) for x in fast_iterators_raw)
        fast_iterators_transformed = np.zeros((len(iterators), 
                                               max_word_length, max_dim + 1))
        scales = np.zeros((len(fast_iterators_raw),))
        for i in range(len(fast_iterators_raw)):
            scales[i] = fast_iterators[i].scale
            for j in range(len(fast_iterators_raw[i])):
                for k in range(len(fast_iterators_raw[i][j])):
                    fast_iterators_transformed[i,j,k] = \
                                                fast_iterators_raw[i][j][k]
        ISS_fast = _fast_ISS(Z, fast_iterators_transformed, scales)

    # get solution for SummationIterators that are not of type SimpleWord
    if slow_iterators:
        ISS_slow = _slow_ISS(Z, slow_iterators)

    # concatenate results if slow and fast iterators were specified
    if fast_iterators and slow_iterators:
        results = np.zeros((Z.shape[0], len(iterators), Z.shape[2]))
        for i, index in enumerate(fast_iterators_index):
            results[:, index, :] = ISS_fast[:, i, :]
        for i, index in enumerate(slow_iterators_index):
            results[:, index, :] = ISS_slow[:, i, :]
        return results
    elif fast_iterators:
        return ISS_fast
    elif slow_iterators:
        return ISS_slow

@numba.njit(parallel=True, fastmath=True)
def _fast_ISS(Z: np.ndarray, 
              iterators: np.ndarray,
              scales: np.ndarray) -> np.ndarray:
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

def _slow_ISS(Z: np.ndarray,
              iterators: list) -> np.ndarray:
    result = np.zeros((Z.shape[0], len(iterators), Z.shape[2]))
    for i in range(Z.shape[0]):
        for j in range(len(iterators)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            for k, mon in enumerate(iterators[j].monomials()):
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(mon)):
                    C = C * mon[l](Z[i, :, :])
                result[i, j, :] = np.cumsum(result[i, j, :] * C)
                result[i, j, :] /= Z.shape[2]**iterators[j].scale

    return result
