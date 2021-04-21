import numpy as np

class FeatureFilter:
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
		return "FeatureFilter('"+self._name+"')"
	
	def __call__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs
		return self
		
	def filter(self, X:np.ndarray):
		if self._func is None:
			raise RuntimeError("No function specified")
		X = np.atleast_2d(X)
		return self._func(X, *self._args, **self._kwargs)

def _ppv(X:np.ndarray,
		 quantile:float=0.5,
		 constant:bool=False) -> np.ndarray:
	if constant:
		ref_value = quantile
	else:
		if not 0<quantile<1:
			raise ValueError("If 'constant' is set to False, quantile has "
							 "to be a value between 0 and 1")
		selection = np.random.choice(np.arange(len(X)), size=20)
		ref_value = np.quantile(np.array([X[i] for i in selection]).flatten(),
								quantile)
	result = np.zeros(len(X))
	for i in range(len(X)):
		result[i] = np.sum(X[i]>=ref_value)/len(X[i])
	return result

PPV = FeatureFilter("proportion of positive values")
PPV.set_function(_ppv)

def _max(X:np.ndarray):
	return np.max(X, axis=1)

MAX = FeatureFilter("maximal value")
MAX.set_function(_max)

def _min(X:np.ndarray):
	return np.min(X, axis=1)

MIN = FeatureFilter("minimal value")
MIN.set_function(_min)