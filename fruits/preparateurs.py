import numpy as np

class DataPreparateur:
	def __init__(self, name:str):
		self._name = name
		self._func = None

	def set_function(self, func):
		if not callable(func):
			raise TypeError("DataPreparateur needs to have a callable function")
		self._func = func

	@property
	def name(self) -> str:
		return self._name

	@name.setter
	def name(self, name:str):
		self._name = name

	def __str__(self) -> str:
		return "DataPreparateur: "+self._name

	def __call__(self, X:np.ndarray):
		if self._func is None:
			raise RuntimeError("No function specified")
		return self._func(X)

ID = DataPreparateur("identity function")
ID.set_function(lambda x: x)

def _inc(X:np.ndarray) -> np.ndarray:
	out = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
	pad_widths = [(0,0) for dim in range(X.ndim)]
	pad_widths[2] = (1,0)
	out = np.pad(out, pad_width=pad_widths, mode="constant")
	return out

INC = DataPreparateur("increments")
INC.set_function(_inc)