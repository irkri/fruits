import numpy as np

class Feature:
	''' an instance of this class takes all time series data as input and 
	returns for each time series the value of one feature '''
	def __init__(self, name:str=""):
		self._name = name
		self._func = None

	def set_function(self, f):
		if not callable(f):
			raise TypeError("Feature needs to have a callable function")
		self._func = f

	@property
	def name(self) -> str:
		return self._name

	@name.setter
	def name(self, name:str):
		self._name = name

	def __str__(self) -> str:
		return "Feature: "+self._name
	
	def __call__(self, X:np.ndarray, *args, **kwargs):
		''' feature needs to be called on all input data '''
		if X.ndim<2:
			raise ValueError("Data has to be at least 2-dimesional")
		if self._func is None:
			raise RuntimeError("No function specified")
		return self._func(X, *args, **kwargs)

def _ppv(X:np.ndarray) -> np.ndarray:
	# one feature for every time series in X
	result = np.zeros(len(X))
	# take a random sample of the dataset and compute the median as reference
	# value
	rand_ind = np.random.choice(np.arange(len(X)), size=20)
	med = np.median(np.array([X[i] for i in rand_ind]).flatten())
	for i in range(len(X)):
		result[i] = np.sum(X[i]>=med)/len(X[i])
	return result

PPV = Feature("proportion of positive values")
PPV.set_function(_ppv)