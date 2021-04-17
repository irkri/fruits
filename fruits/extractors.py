import numpy as np
import re
from fruits.iterators import SummationIterator, Word
from fruits.preparateurs import DataPreparateur
from fruits.features import Feature

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

class FeatureExtractor:
	''' class that connects all different options the user can choose from,
	e.g. SummationIterator, used functions to extract features, ... '''
	def __init__(self):
		# used functions and class instances for data processing
		# preparateurs will be called in the order that they're added
		self._preparateurs = []
		self._iterators = []
		self._features = []

		# bool variables so data has to be processed only once
		self._prepared = False
		self._iterated = False
		self._featured = False

		# data variables that hold the processed data
		self._prepared_data = None
		self._iterated_data = None
		self._featured_data = None

		# first step at every action with FeatureExtractor: set input data
		self._input_data = None
		# shape of input data
		self._ts = 0
		self._ts_dim = 0
		self._ts_length = 0

	def nfeatures(self) -> int:
		return len(self._features)*len(self._iterators)

	def add_data_preparateur(self, preparateur:DataPreparateur):
		if not isinstance(preparateur, DataPreparateur):
			raise TypeError
		self._preparateurs.append(preparateur)
		self._prepared = False
		self._iterated = False
		self._featured = False

	def clear_preparateurs(self):
		self._preparateurs = []

	def add_summation_iterator(self, sum_iter:SummationIterator):
		if not isinstance(sum_iter, SummationIterator):
			raise TypeError
		self._iterators.append(sum_iter)
		self._iterated = False
		self._featured = False

	def clear_iterators(self):
		self._iterators = []
		
	def add_feature(self, feat:Feature):
		if not isinstance(feat, Feature):
			raise TypeError
		self._features.append(feat)
		self._featured = False

	def clear_features(self):
		self._features = []

	def add(self, obj):
		if isinstance(obj, DataPreparateur):
			self.add_data_preparateur(obj)
		elif isinstance(obj, SummationIterator):
			self.add_summation_iterator(obj)
		elif isinstance(obj, Feature):
			self.add_feature(obj)
		else:
			raise TypeError(f"Cannot add variable of type {type(obj)}")

	def clear(self):
		self.clear_preparateurs()
		self.clear_iterators()
		self.clear_features()

	def set_input_data(self, X:np.ndarray):
		if len(X.shape)==1:
			X = np.expand_dims(X, axis=0)
		if len(X.shape)==2:
			X = np.expand_dims(X, axis=1)
		if X.ndim!=3:
			raise ValueError("Unsupported data dimensionality")
		self._ts = X.shape[0]
		self._ts_dim = X.shape[1]
		self._ts_length = X.shape[2]
		self._input_data = X
		self._prepared = False
		self._iterated = False
		self._featured = False

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

	def feature(self):
		if self._featured:
			return
		self.prepare()
		self.iterate()
		if not self._features:
			raise RuntimeError("No Feature specified")
		self._featured_data = np.zeros((self._ts, self.nfeatures()))
		k = 0
		for i in range(len(self._iterators)):
			for j, feat in enumerate(self._features):
				self._featured_data[:, k] = feat(self._iterated_data[:, i, :])
				k += 1
		self._featured = True

	def features(self) -> np.ndarray:
		self.feature()
		return self._featured_data

	def __call__(self, Z:np.ndarray) -> np.ndarray:
		self.set_input_data(Z)
		return self.features()