import numpy as np
import re
from fruits.iterators import SummationIterator, Word

def iterated_sums(Z:np.ndarray, 
				  sum_iter:SummationIterator) -> np.ndarray:
	''' calculates the iterated sums signature for a given input series Z,
	e.g. <[1][12][22],ISS(Z)> = CS(CS(CS(Z[0])*Z[0]*Z[1])*Z[1]^2)
	(CS is np.cumsum) '''
	if len(Z.shape)==1:
		Z = np.expand_dims(Z, axis=0)
	length = Z.shape[0]
	P = np.ones(length, dtype=np.float64)
	for word in sum_iter.words():
		P = np.cumsum(P*word(Z))
	return P

def get_increments(X:np.ndarray, axis=0) -> np.ndarray:
	''' returns array of increments x_i - x_{i-1} '''
	out = np.delete((np.roll(X, -1, axis=axis) - X), -1, axis=axis)
	pad_widths = [(0,0) for dim in range(X.ndim)]
	pad_widths[axis] = (1,0)
	out = np.pad(out, pad_width=pad_widths, mode="constant")
	return out