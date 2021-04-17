import numpy as np
import re
from fruits.iterators import SummationIterator, Word
from fruits.preparateurs import DataPreparateur
from fruits.features import FeatureFilter

def iterated_sums(Z:np.ndarray, 
				  sum_iter:SummationIterator) -> np.ndarray:
	''' calculates the iterated sums signature for a given input series Z,
	e.g. <[1][12][22],ISS(Z)> = CS(CS(CS(Z[0])*Z[0]*Z[1])*Z[1]^2)
	(CS is np.cumsum) '''
	if len(Z.shape)==1:
		Z = np.expand_dims(Z, axis=0)
	P = np.ones(Z.shape[1], dtype=np.float64)
	for word in sum_iter.words():
		P = np.cumsum(P*word(Z))
	return P	

class Fruit:
	''' class that connects all different options the user can choose from,
	e.g. SummationIterator, used functions to extract features, ... '''
	def __init__(self):
		# used functions and class instances for data processing
		# preparateurs will be called in the order that they're added
		self._preparateurs = []
		self._iterators = []
		self._filters = []

		# bool variables so data has to be processed only once
		self._prepared = False
		self._iterated = False
		self._filtered = False

		# data variables that hold the processed data
		self._prepared_data = None
		self._iterated_data = None
		self._filtered_data = None

		# first step at every action with FeatureFilter: set input data
		self._input_data = None
		# shape of input data
		self._ts = 0
		self._ts_dim = 0
		self._ts_length = 0

	def nfeatures(self) -> int:
		return len(self._filters)*len(self._iterators)

	def add_data_preparateur(self, preparateur:DataPreparateur):
		if not isinstance(preparateur, DataPreparateur):
			raise TypeError
		self._preparateurs.append(preparateur)
		self._prepared = False
		self._iterated = False
		self._filtered = False

	def clear_data_preparateurs(self):
		self._preparateurs = []
		self._prepared = False

	def add_summation_iterator(self, sum_iter:SummationIterator):
		if not isinstance(sum_iter, SummationIterator):
			raise TypeError
		self._iterators.append(sum_iter)
		self._iterated = False
		self._filtered = False

	def clear_summation_iterators(self):
		self._iterators = []
		self._iterated = False
		
	def add_feature_filter(self, feat:FeatureFilter):
		if not isinstance(feat, FeatureFilter):
			raise TypeError
		self._filters.append(feat)
		self._filtered = False

	def clear_feature_filter(self):
		self._filters = []
		self._filtered = False

	def add(self, obj):
		if isinstance(obj, DataPreparateur):
			self.add_data_preparateur(obj)
		elif isinstance(obj, SummationIterator):
			self.add_summation_iterator(obj)
		elif isinstance(obj, FeatureFilter):
			self.add_feature_filter(obj)
		else:
			raise TypeError(f"Cannot add variable of type {type(obj)}")

	def clear(self):
		self.clear_data_preparateurs()
		self.clear_summation_iterators()
		self.clear_feature_filters()

	def set_input_data(self, X:np.ndarray):
		if len(X.shape)==1:
			X = np.expand_dims(X, axis=0)
		if len(X.shape)==2:
			X = np.expand_dims(X, axis=1)
		if X.ndim!=3:
			raise ValueError("Unsupported input shape")
		self._ts = X.shape[0]
		self._ts_dim = X.shape[1]
		self._ts_length = X.shape[2]
		self._input_data = X
		self._prepared = False
		self._iterated = False
		self._filtered = False

	def prepare(self):
		if self._input_data is None:
			raise RuntimeError("No input data")
		if self._prepared:
			return
		self._prepared_data = self._input_data
		for prep in self._preparateurs:
			self._prepared_data = prep(self._prepared_data)
		self._prepared = True

	def prepared_data(self) -> np.ndarray:
		self.prepare()
		return self._prepared_data

	def iterate(self):
		if self._iterated:
			return
		if not self._iterators:
			raise RuntimeError("No SummationIterator specified")
		self.prepare()
		self._iterated_data = np.zeros((self._ts, len(self._iterators), 
										self._ts_length))
		for i in range(self._ts):
			for j in range(len(self._iterators)):
				self._iterated_data[i, j, :] = iterated_sums(
							self._prepared_data[i, :, :], self._iterators[j])
		self._iterated = True

	def iterated_sums(self) -> np.ndarray:
		self.iterate()
		return self._iterated_data

	def filter(self):
		if self._filtered:
			return
		self.prepare()
		self.iterate()
		if not self._filters:
			raise RuntimeError("No FeatureFilter specified")
		self._filtered_data = np.zeros((self._ts, self.nfeatures()))
		k = 0
		for i in range(len(self._iterators)):
			for j, feat in enumerate(self._filters):
				self._filtered_data[:, k] = feat(self._iterated_data[:, i, :])
				k += 1
		self._filtered = True

	def features(self) -> np.ndarray:
		self.filter()
		return self._filtered_data

	def __call__(self, Z:np.ndarray) -> np.ndarray:
		self.set_input_data(Z)
		return self.features()