import numpy as np
import re
from fruits.iterators import SummationIterator, Word
import fruits.preparateurs as prep
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
		self._data_preparateur = prep.ID
		self._sum_iter = None
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
		return len(self._features)

	def set_data_preparateur(self, preparateur:prep.DataPreparateur):
		if not isinstance(preparateur, prep.DataPreparateur):
			raise TypeError
		self._data_preparateur = preparateur
		self._prepared = False
		self._iterated = False
		self._featured = False

	def set_summation_iterator(self, sum_iter:SummationIterator):
		if not isinstance(sum_iter, SummationIterator):
			raise TypeError
		self._sum_iter = sum_iter
		self._iterated = False
		self._featured = False
		
	def add_feature(self, feat:Feature):
		if not isinstance(feat, Feature):
			raise TypeError
		self._features.append(feat)
		self._featured = False

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
			raise RuntimeError("No input data set")
		if self._prepared:
			return
		self._prepared_data = self._data_preparateur(self._input_data)
		self._prepared = True

	def prepared_data(self) -> np.ndarray:
		self.prepare()
		return self._prepared_data

	def iterate(self):
		if self._iterated:
			return
		self.prepare()
		iss = np.zeros((self._ts, self._ts_length))
		for i in range(self._ts):
			self._iterated_data[i, :] = iterated_sums(
							self._prepared_data[i, :, :], self._sum_iter)
		self._iterated = True

	def iterated_sums(self) -> np.ndarray:
		self.iterate()
		return self._iterated_data

	def feature(self):
		if self._featured:
			return
		self.prepare()
		self.iterate()
		self._featured_data = np.zeros((self._ts, self.nfeatures()))
		for i, feat in enumerate(self._features):
			self._featured_data[:, i] = feat(self._iterated_data)
		self._featured = True

	def features(self) -> np.ndarray:
		self.feature()
		return self._featured_data

	def __call__(self, Z:np.ndarray) -> np.ndarray:
		self.set_input_data(Z)
		return self.features()