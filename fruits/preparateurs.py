import numpy as np

class DataPreparateur:
	"""Class DataPreperateur
	
	A DataPreparateur object can be called on a three dimensional numpy 
	array. The output should be a numpy array that matches the shape of 
	the input array.
	"""
	def __init__(self, name:str):
		self._name = name
		self._func = None

	def set_function(self, func):
		if not callable(func):
			raise TypeError("Cannot set non-callable object as function")
		self._func = func

	@property
	def name(self) -> str:
		return self._name

	@name.setter
	def name(self, name:str):
		self._name = name

	def __repr__(self) -> str:
		return "DataPreparateur('"+self._name+"')"

	def __call__(self, X:np.ndarray):
		if self._func is None:
			raise RuntimeError("No function specified")
		X = np.atleast_3d(X)
		return self._func(X)

def _inc(X:np.ndarray) -> np.ndarray:
	out = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
	pad_widths = [(0,0) for dim in range(X.ndim)]
	pad_widths[2] = (1,0)
	out = np.pad(out, pad_width=pad_widths, mode="constant")
	return out

INC = DataPreparateur("increments")
INC.set_function(_inc)

def _std(X:np.ndarray) -> np.ndarray:
	out = np.zeros(X.shape)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			out[i, j, :] = (X[i, j, :]-np.mean(X[i, j, :]))/np.std(X[i, j, :])
	return out

STD = DataPreparateur("standardization")
STD.set_function(_std)