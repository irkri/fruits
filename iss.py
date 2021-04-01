import numpy as np
from timeit import default_timer as timer

class CIterator:
	def __init__(self, container):
		self._container = container._content
		self._index = 0

	def __next__(self):
		if self._index<len(self._container):
			result = self._container[self._index]
			self._index += 1
			return result
		raise StopIteration

class Composition:
	def __init__(self, in_string:str):
		in_string = in_string.replace("[","").replace("]","")
		self._content = [int(x) for x in in_string]

	def __mul__(self, other):
		self._content.append(other)

	def __iter__(self):
		return CIterator(self)

	def __str__(self):
		return "["+"".join([str(x) for x in self._content])+"]"

class Concatination:
	def __init__(self):
		self._content = []

	def __mul__(self, other):
		self._content.append(other)
		return self

	def __str__(self):
		return "".join([str(x) for x in self._content])

	def __iter__(self):
		return CIterator(self)

	@staticmethod
	def from_str(string:str):
		conc = Concatination()
		elements = [x[1:] for x in string.split("]")][:-1]
		for element in elements:
			conc *= Composition(element)
		return conc

def iterated_sums(Z:np.array, 
				  concatination:Concatination, 
				  verbose:bool=False) -> np.array:
	''' calculates the iterated sums signature for a given input series Z,
	e.g. <[1][12][22],ISS(Z)> = CS(CS(CS(Z[0])*Z[0]*Z[1])*Z[1]^2)
	(CS is np.cumsum) '''
	if verbose:
		print("{:=^80}".format("Iterated Sums Signature"))
		print(f"Input: <{concatination},ISS(Z)>\nwith Z={Z}\n")
		_start = timer()
	if len(Z.shape)==1:
		Z = Z.reshape(len(Z),1)
	length = Z.shape[0]
	P = np.ones(length, dtype=np.float64)
	for element in concatination:
		C = np.ones(length, dtype=np.float64)
		for letter in element:
			C *= Z[:,int(letter)-1]
		P = np.cumsum(P*C)
		if verbose:
			print("{:-^40}".format("Start Iteration"))
			print(f"Composition: {element}")
			print(f"P: {P}")
			print(f"C: {C}")
	if verbose:
		print("\nDone.")
		print("Time needed: {:.5f}s".format(timer()-_start))
		print("{:=^80}".format(""))
	return P

def features_from_iterated_sums(Z_array:np.array,
								concatinations:list):
	_features = np.zeros((Z_array.shape[0], len(concatinations)))
	for i in range(Z_array.shape[0]):
		for j in range(len(concatinations)):
			_features[i,j] = ppv(iterated_sums(Z_array[i], concatinations[j]))
	return _features

def generate_concatinations(number:int, dim:int=1,
							max_composition_length:int=5,
							max_concatination_length:int=10):
	concs = []
	av_elements = [str(i+1) for i in range(dim)]
	for i in range(number):
		conc = Concatination()
		length = np.random.randint(1,max_concatination_length+1)
		for j in range(length):
			clength = np.random.randint(1,max_composition_length+1)
			conc *= Composition("".join(np.random.choice(av_elements, 
														size=clength)))
		concs.append(conc)

	return concs

def get_increments(X:np.array, axis=0) -> np.array:
	''' returns array of increments x_i - x_{i-1} '''
	out = np.delete((np.roll(X, -1, axis=axis) - X), -1, axis=axis)
	pad_widths = [(0,0) for dim in range(X.ndim)]
	pad_widths[axis] = (1,0)
	out = np.pad(out, pad_width=pad_widths, mode="constant")
	return out

def ppv(X:np.array) -> float:
	if len(X)==0:
		return 0
	return np.sum(X>=0)/len(X)