import re

import numpy as np

from fruits.core import ISS
from fruits.features import FeatureSieve
from fruits.iterators import SummationIterator
from fruits.preparateurs import DataPreparateur

class Fruit:
    """Feature Extractor using iterated sums.
    
    A Fruit object consists of a number of FruitBranch objects. At the
    end of the pipeline, each branch returns their own features and
    they will be concatenated by this class.
    """
    def __init__(self, name: str = ""):
        # simple identifier for the Fruit object
        self.name = name
        # list of FruitBranches
        self._branches = [FruitBranch()]
        # pointer for the current branch
        self._current_branch = self._branches[0]
        self._current_branch_index = 0
        # arrays of processed data for each branch
        # if only one branch exists, these arrays will be three dimensional
        # for multiple branches they are going to have four dimensions
        self._prepared_data = None
        self._iterated_data = None
        self._sieved_data = None

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name:str):
        self._name = name

    def add_branch(self, branch):
        """Adds a new branch to the pipeline.

        :param branch: new branch
        :type branch: FruitBranch
        """
        self._branches.append(branch)

    def start_new_branch(self):
        """Adds a new and empty branch to the pipeline and switches to
        it. All future operations on this Fruit object will be called
        on the new branch.
        """
        self._branches.append(FruitBranch())
        self._current_branch = self._branches[-1]
        self._current_branch_index = len(self._branches) - 1

    def branches(self) -> list:
        """Returns all branches of this Fruit object.
        
        :returns: list of branches
        :rtype: {list}
        """
        return self._branches

    def switch_branch(self, index: int):
        """Switches to the branch with the given index. If the Fruit
        objects points to a branch, then each method that is called on
        the Fruit object is actually called on this branch.
        
        :param index: index between 0 and len(self.branches)
        :type index: int
        """
        self._current_branch = self._branches[index]
        self._current_branch_index = index

    def current_branch(self):
        """Returns the branch that is currently selected.

        The selected branch is used to extend the pipeline by calling
        functions like `add()` on the Fruit object. Normally this is
        done on a FruitBranch object. 
        
        :returns: the current branch in self.branches
        :rtype: FruitBranch
        """
        return self._current_branch_index

    def current_branch_index(self) -> int:
        """Returns the index of the currently selected branch.
        
        :returns: index of the current branch in self.branches
        :rtype: int
        """
        return self._current_branch_index

    def nfeatures(self) -> int:
        """Returns the total number of features of all branches 
        combined.
        
        :returns: number of features
        :rtype: {int}
        """
        return sum([branch.nfeatures() for branch in self._branches])

    def add_data_preparateur(self, preparateur: DataPreparateur):
        self._current_branch.add_data_preparateur(preparateur)

    def get_data_preparateurs(self) -> list:
        return self._current_branch.get_data_preparateurs()

    def clear_data_preparateurs(self):
        self._current_branch.clear_data_preparateurs()

    def add_summation_iterator(self, sum_iter: SummationIterator):
        self._current_branch.add_summation_iterator(sum_iter)

    def get_summation_iterators(self) -> list:
        return self._current_branch.get_summation_iterators()

    def clear_summation_iterators(self):
        self._current_branch.clear_summation_iterators()
        
    def add_feature_sieve(self, feat: FeatureSieve):
        self._current_branch.add_feature_sieve(feat)

    def get_feature_sieves(self) -> list:
        return self._current_branch.get_feature_sieves()

    def clear_feature_sieves(self):
        self._current_branch.clear_feature_sieves()

    def add(self, *objects):
        self._current_branch.add(objects)

    def clear(self):
        self._current_branch.clear()

    def set_input_data(self, X: np.ndarray):
        for branch in self._branches:
            branch.set_input_data(X)

    def prepared_data(self) -> np.ndarray:
        """Returns the prepared data of the current branch.
        
        :returns: prepared data array
        :rtype: {np.ndarray}
        """
        return self._current_branch.prepared_data()

    def iterated_sums(self) -> np.ndarray:
        """Returns the iterated sums of the current branch.
        
        :returns: three dimensional array of iterated sums
        :rtype: {np.ndarray}
        """
        return self._current_branch.iterated_sums()

    def sieved_data(self):
        """Returns the features of the current branch.
        
        :returns: feature array
        :rtype: {np.ndarray}
        """
        return self._current_branch.features()

    def features(self) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        branches this Fruit object contains.

        :returns: two dimensional feature array
        :rtype: {np.ndarray}
        """
        if len(self._branches) == 1:
            return self._current_branch.features()
        result = np.zeros((self._current_branch._ts, self.nfeatures()))
        index = 0
        for branch in self._branches:
            k = branch.nfeatures()
            result[:, index:index+k] = branch.features()
            index += k
        return result   

    def copy(self):
        copy_ = Fruit(self.name+" (Copy)")
        copy_._branches = [self._current_branch.copy()]
        copy_._current_branch = copy_._branches[0]
        for i in range(len(self._branches)):
            if i != self._current_branch_index:
                copy_.add_branch(self._branches[i])
        return copy_

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.set_input_data(X)
        return self.features()

    def __copy__(self):
        return self.copy()


class FruitBranch:
    """One branch for a Fruit object.
    
    A FruitBranch object extracts values from time series data that are 
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
    the iterated sums from the previous step. The Fruit object then  
    returns an array of numbers, i.e. the features for each time series.
    The number of features for one time series is equal to:: 
        [number of iterators] x [number of sieves] 
    """
    def __init__(self):
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

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.
        
        :returns: number of features
        :rtype: {int}
        """
        return len(self._sieves) * len(self._iterators)

    def add_data_preparateur(self, preparateur: DataPreparateur):
        if not isinstance(preparateur, DataPreparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._prepared = False
        self._iterated = False
        self._sieved = False

    def get_data_preparateurs(self) -> list:
        return self._preparateurs

    def clear_data_preparateurs(self):
        self._preparateurs = []
        self._prepared = False
        self._iterated = False
        self._sieved = False

    def add_summation_iterator(self, sum_iter: SummationIterator):
        if not isinstance(sum_iter, SummationIterator):
            raise TypeError
        self._iterators.append(sum_iter)
        self._iterated = False
        self._sieved = False

    def get_summation_iterators(self) -> list:
        return self._iterators

    def clear_summation_iterators(self):
        self._iterators = []
        self._iterated = False
        self._sieved = False
        
    def add_feature_sieve(self, feat: FeatureSieve):
        if not isinstance(feat, FeatureSieve):
            raise TypeError
        self._sieves.append(feat)
        self._sieved = False

    def get_feature_sieves(self) -> list:
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

    def set_input_data(self, X: np.ndarray):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
        if X.ndim != 3:
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

    def copy(self):
        copy_ = FruitBranch()
        for preparateur in self.get_data_preparateurs():
            copy_.add(preparateur)
        for iterator in self.get_summation_iterators():
            copy_.add(iterator)
        for sieve in self.get_feature_sieves():
            copy_.add(sieve)
        return copy_

    def __call__(self, Z:np.ndarray) -> np.ndarray:
        self.set_input_data(Z)
        return self.features()

    def __copy__(self):
        return self.copy()