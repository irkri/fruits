import numpy as np
import re
from fruits.iterators import SummationIterator
from fruits.preparateurs import DataPreparateur
from fruits.features import FeatureSieve
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
	FeatureSieves may now be added to the Fruit object.
	Each sieve has a corresponding function that will be called on 
	the	iterated sums from the previous step. The Fruit object then  
	returns an array of numbers, i.e. the features for each time series.
	The number of features for one time series is equal to:: 
		[number of iterators] x [number of sieves] 
	"""
	def __init__(self, name=""):
		# simple identifier for the Fruit object
		self.name = name
		# used functions and class instances for data processing
		# preparateurs will be called in the order that they're added
		self._preparateurs = []
		self._iterators = []
		self._sieves = []

		# bool variables so data has to be processed only once
		self._prepared = False
		self._iterated = False
		self._sieved = False

		# data variables that hold the processed data
		self._prepared_data = None
		self._iterated_data = None
		self._sieved_data = None

		# first step at every action with Fruit: set input data
		self._input_data = None
		# shape of input data
		self._ts = 0
		self._ts_dim = 0
		self._ts_length = 0

	@property
	def name(self) -> str:
		return self._name
	
	@name.setter
	def name(self, name:str):
		self._name = name

	def nfeatures(self) -> int:
		"""Returns the total number of features the current
		configuration produces.
		
		:returns: number of features
		:rtype: {int}
		"""
		return len(self._sieves)*len(self._iterators)

	def add_data_preparateur(self, preparateur:DataPreparateur):
		if not isinstance(preparateur, DataPreparateur):
			raise TypeError
		self._preparateurs.append(preparateur)
		self._prepared = False
		self._iterated = False
		self._sieved = False

	def get_data_preparateurs(self):
		return self._preparateurs

	def clear_data_preparateurs(self):
		self._preparateurs = []
		self._prepared = False
		self._iterated = False
		self._sieved = False

	def add_summation_iterator(self, sum_iter:SummationIterator):
		if not isinstance(sum_iter, SummationIterator):
			raise TypeError
		self._iterators.append(sum_iter)
		self._iterated = False
		self._sieved = False

	def get_summation_iterators(self):
		return self._iterators

	def clear_summation_iterators(self):
		self._iterators = []
		self._iterated = False
		self._sieved = False
		
	def add_feature_sieve(self, feat:FeatureSieve):
		if not isinstance(feat, FeatureSieve):
			raise TypeError
		self._sieves.append(feat)
		self._sieved = False

	def get_feature_sieves(self):
		return self._sieves

	def clear_feature_sieves(self):
		self._sieves = []
		self._sieved = False

	def add(self, *objects):
		objects = np.array(objects, dtype=object).flatten()
		for obj in objects:
			if isinstance(obj, DataPreparateur):
				self.add_data_preparateur(obj)
			elif isinstance(obj, SummationIterator):
				self.add_summation_iterator(obj)
			elif isinstance(obj, FeatureSieve):
				self.add_feature_sieve(obj)
			else:
				raise TypeError(f"Cannot add variable of type {type(obj)}")

	def clear(self):
		self.clear_data_preparateurs()
		self.clear_summation_iterators()
		self.clear_feature_sieves()

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
		self._sieved = False

	def _prepare(self):
		if self._input_data is None:
			raise RuntimeError("No input data")
		if self._prepared:
			return
		self._prepared_data = self._input_data
		for prep in self._preparateurs:
			self._prepared_data = prep.prepare(self._prepared_data)
		self._prepared = True

	def prepared_data(self) -> np.ndarray:
		self._prepare()
		return self._prepared_data

	def _iterate(self):
		if self._iterated:
			return
		if not self._iterators:
			raise RuntimeError("No SummationIterator specified")
		self._prepare()
		self._iterated_data = np.zeros((self._ts, len(self._iterators), 
										self._ts_length))
		self._iterated_data = ISS(self._prepared_data, self._iterators)
		self._iterated = True

	def iterated_sums(self) -> np.ndarray:
		self._iterate()
		return self._iterated_data

	def _sieve(self):
		if self._sieved:
			return
		self._iterate()
		if not self._sieves:
			raise RuntimeError("No FeatureSieve specified")
		self._sieved_data = np.zeros((self._ts, self.nfeatures()))
		k = 0
		for i in range(len(self._iterators)):
			for feat in self._sieves:
				self._sieved_data[:, k] = feat.sieve(
					self._iterated_data[:, i, :])
				k += 1
		self._sieved = True

	def features(self) -> np.ndarray:
		self._sieve()
		return self._sieved_data

	def __call__(self, Z:np.ndarray) -> np.ndarray:
		self.set_input_data(Z)
		return self.features()