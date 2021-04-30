import numpy as np
import numba
from fruits.iterators import SummationIterator, SimpleWord

BREAK_ITERATION_IDENTIFIER = -1

def ISS(Z:np.ndarray, iterators:list) -> np.ndarray:
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
	fast_iterators = []
	slow_iterators = []
	for iterator in iterators:
		if isinstance(iterator, SimpleWord):
			fast_iterators.append(iterator)
		elif isinstance(iterator, SummationIterator):
			slow_iterators.append(iterator)
		else:
			raise TypeError("Iterators should be of type SummationIterator")

	if fast_iterators:
		max_dim = max(x.max_dim for x in fast_iterators)
		fast_iterators = [list(x.monomials()) for x in fast_iterators]
		max_word_length = max(len(x) for x in fast_iterators)
		fast_iterators_transformed = np.zeros((len(iterators), 
											   max_word_length, max_dim+1))
		# fast_iterators_transformed += BREAK_ITERATION_IDENTIFIER
		for i in range(len(fast_iterators)):
			for j in range(len(fast_iterators[i])):
				for k in range(len(fast_iterators[i][j])):
					fast_iterators_transformed[i,j,k] = fast_iterators[i][j][k]
		ISS_fast = _fast_ISS(Z, fast_iterators_transformed)

	if slow_iterators:
		ISS_slow = _slow_ISS(Z, slow_iterators)

	if fast_iterators and slow_iterators:
		return np.vstack(ISS_fast, ISS_slow)
	elif fast_iterators:
		return ISS_fast
	elif slow_iterators:
		return ISS_slow

@numba.njit(parallel=True, fastmath=True)
def _fast_ISS(Z:np.ndarray, iterators:np.ndarray) -> np.ndarray:
	result = np.zeros((Z.shape[0], len(iterators), Z.shape[2]))
	for i in numba.prange(Z.shape[0]):
		for j in numba.prange(len(iterators)):
			result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
			for k in numba.prange(len(iterators[j])):
				if not np.any(iterators[j][k]):
					continue
				C = np.ones(Z.shape[2], dtype=np.float64)
				for l in range(len(iterators[j][k])):
					if iterators[j][k][l]!=0:
						C = C * Z[i, l, :]**iterators[j][k][l]
				result[i, j, :] = np.cumsum(result[i, j, :]*C)

	return result

def _slow_ISS(Z:np.ndarray, iterators:list) -> np.ndarray:
	result = np.zeros((Z.shape[0], len(iterators), Z.shape[2]))
	for i in range(Z.shape[0]):
		for j in range(len(iterators)):
			result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
			for k, mon in enumerate(iterators[j].monomials()):
				C = np.ones(Z.shape[2], dtype=np.float64)
				for l in range(len(mon)):
					C = C * mon[l](Z[i, :, :])
				result[i, j, :] = np.cumsum(result[i, j, :]*C)

	return result