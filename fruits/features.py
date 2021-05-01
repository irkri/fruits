import numpy as np
from copy import copy
import numba

class FeatureSieve:
	"""Class FeatureSieve
	
	A FeatureSieve object is used to extract a single number out of an
	multidimensional numpy array.
	"""
	def __init__(self, name:str=""):
		self._name = name
		self._args = ()
		self._kwargs = {}
		self._func = None

	def set_function(self, f):
		if not callable(f):
			raise TypeError("Cannot set non-callable object as function")
		self._func = f

	@property
	def name(self) -> str:
		return self._name

	@name.setter
	def name(self, name:str):
		self._name = name

	def __repr__(self) -> str:
		out = "FeatureSieve('"+self._name+"'"
		if self._args:
			out += ",".join(self._args)
		if self._kwargs:
			out += ","+",".join([str(x)+"="+str(self._kwargs[x]) 
								 for x in self._kwargs])
		out += ")"
		return out
	
	def __call__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs
		return copy(self)

	def __copy__(self):
		ff = FeatureSieve(self.name)
		ff._args = self._args
		ff._kwargs = self._kwargs
		self._args = ()
		self._kwargs = {}
		ff.set_function(self._func)
		return ff
		
	def sieve(self, X:np.ndarray):
		if self._func is None:
			raise RuntimeError("No function specified")
		X = np.atleast_2d(X)
		return self._func(X, *self._args, **self._kwargs)

@numba.njit(parallel=True, fastmath=True)
def _fast_ppv(X:np.ndarray, ref_value:float) -> np.ndarray:
	result = np.zeros(len(X))
	for i in numba.prange(len(X)):
		c = 0
		for j in range(len(X[i])):
			if X[i][j]>=ref_value:
				c += 1
		if len(X[i])==0:
			result[i] = 0
		else:
			result[i] = c/len(X[i])
	return result

def _ppv(X:np.ndarray,
		 quantile:float=0.5,
		 constant:bool=False,
		 sample_size:float=0.05) -> np.ndarray:
	if constant:
		ref_value = quantile
	else:
		if not 0<quantile<1:
			raise ValueError("If 'constant' is set to False, quantile has "+
							 "to be a value between 0 and 1")
		sample_size = int(sample_size*len(X))
		sample_size = 1 if sample_size<1 else sample_size
		selection = np.random.choice(np.arange(len(X)), size=sample_size,
									 replace=False)
		ref_value = np.quantile(np.array([X[i] for i in selection]).flatten(),
								quantile)
	if len(X)==0:
		return 0
	return _fast_ppv(X, ref_value)

PPV = FeatureSieve("proportion of positive values")
PPV.set_function(_ppv)

@numba.njit(parallel=True, fastmath=True)
def _max(X:np.ndarray):
	result = np.zeros(len(X))
	for i in numba.prange(len(X)):
		if len(X[i])==0:
			continue
		maximum = X[i][0]
		for j in range(len(X[i])):
			if X[i][j]>maximum:
				maximum = X[i][j]
		result[i] = maximum
	return result

MAX = FeatureSieve("maximal value")
MAX.set_function(_max)

@numba.njit(parallel=True, fastmath=True)
def _min(X:np.ndarray):
	result = np.zeros(len(X))
	for i in numba.prange(len(X)):
		if len(X[i])==0:
			continue
		minimum = X[i][0]
		for j in range(len(X[i])):
			if X[i][j]<minimum:
				minimum = X[i][j]
		result[i] = minimum
	return result

MIN = FeatureSieve("minimal value")
MIN.set_function(_min)