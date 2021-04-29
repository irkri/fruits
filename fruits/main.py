import numpy as np
import re
from fruits.iterators import SummationIterator
from fruits.preparateurs import DataPreparateur
from fruits.features import FeatureFilter
from fruits.core import ISS

class Fruit:
	"""Feature Extractor using iterated sums.
	
	A Fruit object extracts values from time series data that are 
	somehow representative of the input data.
	The user can customize any of the following three steps the 
	extractor is going to do in order to get the so called features.

	Data Preparation:
	Apply functions at the start of the extraction procedure.
	There are many so called DataPreparateurs in fruits available
	for preprocessing. The preparateurs will be applied sequentially 
	to the input data.

	Calculating Iterated Sums:
	The preprocessed data is now used to calculate the iterated sums
	signature for different SummationIterators the user can specify.
	Multiple iterators lead to multiple data arrays created for one 
	input array.

	Extracting the Features:
	FeatureFilters may now be added to the Fruit object.
	Each filter has a corresponding function that will be called on 
	the	iterated sums from the previous step. The Fruit object then  
	returns an array of numbers, i.e. the features for each time series.
	The number of features for one time series is equal to:: 
		[number of iterators] x [number of filters] 
	"""
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

	def get_data_preparateurs(self):
		return self._preparateurs

	def clear_data_preparateurs(self):
		self._preparateurs = []
		self._prepared = False

	def add_summation_iterator(self, sum_iter:SummationIterator):
		if not isinstance(sum_iter, SummationIterator):
			raise TypeError
		self._iterators.append(sum_iter)
		self._iterated = False
		self._filtered = False

	def get_summation_iterators(self):
		return self._iterators

	def clear_summation_iterators(self):
		self._iterators = []
		self._iterated = False
		
	def add_feature_filter(self, feat:FeatureFilter):
		if not isinstance(feat, FeatureFilter):
			raise TypeError
		self._filters.append(feat)
		self._filtered = False

	def get_feature_filters(self):
		return self._filters

	def clear_feature_filters(self):
		self._filters = []
		self._filtered = False

	def add(self, *objects):
		objects = np.array(objects, dtype=object).flatten()
		for obj in objects:
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
		self._iterated_data = ISS(self._prepared_data, self._iterators)
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
				self._filtered_data[:, k] = feat.filter(
					self._iterated_data[:, i, :])
				k += 1
		self._filtered = True

	def features(self) -> np.ndarray:
		self.filter()
		return self._filtered_data

	def __call__(self, Z:np.ndarray) -> np.ndarray:
		self.set_input_data(Z)
		return self.features()