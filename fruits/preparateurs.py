import numpy as np
from copy import copy

class DataPreparateur:
	"""Class DataPreperateur
	
	A DataPreparateur object can be called on a three dimensional numpy 
	array. The output should be a numpy array that matches the shape of 
	the input array.
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
		return "DataPreparateur('"+self._name+"')"

	def __call__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs
		return copy(self)

	def __copy__(self):
		dp = DataPreparateur(self.name)
		dp._args = self._args
		dp._kwargs = self._kwargs
		self._args = ()
		self._kwargs = {}
		dp.set_function(self._func)
		return dp

	def prepare(self, X:np.ndarray):
		if self._func is None:
			raise RuntimeError("No function specified")
		X = np.atleast_3d(X)
		return self._func(X, *self._args, **self._kwargs)

def _inc(X:np.ndarray, zero_padding:bool=True) -> np.ndarray:
	if zero_padding:
		out = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
		pad_widths = [(0,0) for dim in range(3)]
		pad_widths[2] = (1,0)
		out = np.pad(out, pad_width=pad_widths, mode="constant")
	else:
		out = np.zeros(X.shape)
		out[:, :, 1:] = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
		out[:, :, 0] = X[:, :, 0]
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