import re
import inspect

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
        self.name = name
        # list of FruitBranches
        self._branches = [FruitBranch()]
        # pointer for the current branch
        self._current_branch = self._branches[0]
        self._current_branch_index = 0

    @property
    def name(self) -> str:
        """Name is a simple identifier for the Fruit object.
        
        :returns: Name of the object
        :rtype: str
        """
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
        :rtype: list
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
        return self._current_branch

    def current_branch_index(self) -> int:
        """Returns the index of the currently selected branch.
        
        :returns: index of the current branch in self.branches
        :rtype: int
        """
        return self._current_branch_index

    def add_preparateur(self, preparateur: DataPreparateur):
        """Adds a DataPreparateur object to the currently selected
        branch.

        :param preparateur: New DataPreparateur
        :type preparateur: DataPreparateur
        """
        self._current_branch.add_preparateur(preparateur)

    def get_preparateurs(self) -> list:
        """Returns all DataPreparateur objects that are added to the
        currently selected branch.
        
        :returns: List of DataPreparateur objects 
        :rtype: list
        """
        return self._current_branch.get_preparateurs()

    def clear_preparateurs(self):
        """Removes all added DataPreparateur objects in the currently
        selected branch.
        """
        self._current_branch.clear_preparateurs()

    def add_iterator(self, sum_iter: SummationIterator):
        """Adds a SummationIterator object to the currently selected
        branch.

        :param preparateur: New SummationIterator
        :type preparateur: SummationIterator
        """
        self._current_branch.add_iterator(sum_iter)

    def get_iterators(self) -> list:
        """Returns all SummationIterator objects that are added to the
        currently selected branch.
        
        :returns: List of SummationIterator objects 
        :rtype: list
        """
        return self._current_branch.get_iterators()

    def clear_iterators(self):
        """Removes all added SummationIterator objects in the currently
        selected branch.
        """
        self._current_branch.clear_iterators()
        
    def add_sieve(self, feat: FeatureSieve):
        """Adds a FeatureSieve object to the currently selected branch.

        :param preparateur: New FeatureSieve
        :type preparateur: FeatureSieve
        """
        self._current_branch.add_feature_sieve(feat)

    def get_sieves(self) -> list:
        """Returns all FeatureSieve objects that are added to the
        currently selected branch.
        
        :returns: List of FeatureSieve objects 
        :rtype: list
        """
        return self._current_branch.get_sieves()

    def clear_sieves(self):
        """Removes all added FeatureSieve objects in the currently
        selected branch.
        """
        self._current_branch.clear_sieves()

    def add(self, *objects):
        """Adds one or multiple object(s) to the currently selected
        branch.
        These objects can be one of the following types:
        - fruits.preparateurs.DataPreparateur
        - fruits.iterators.SummationIterator
        - fruits.features.FeatureSieve
        
        :param objects: Object(s) to add
        :type objects: Object of mentioned type(s) or iterable object
        containing multiple objects of mentioned type(s)
        :raises: TypeError if one of the objects has an unknown type
        """
        self._current_branch.add(objects)

    def clear(self):
        """Clears all DataPreparateur, SummationIterator and
        FeatureSieve objects in the currently selected branch.
        """
        self._current_branch.clear()

    def nfeatures(self) -> int:
        """Returns the total number of features of all branches 
        combined.
        
        :returns: number of features
        :rtype: int
        """
        return sum([branch.nfeatures() for branch in self._branches])

    def fit(self, X: np.ndarray):
        """Fits all branches to the given data.
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        for branch in self._branches:
            branch.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        branches this Fruit object contains.

        :returns: two dimensional feature array
        :rtype: np.ndarray
        :raises: RuntimeError if Fruit.fit wasn't called
        """
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for branch in self._branches:
            k = branch.nfeatures()
            result[:, index:index+k] = branch.transform(X)
            index += k
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits all branches to the given dataset and returns the
        transformed results of X from all branches.
        
        :param X: features of each time series in X for all branches
        concatenated
        :type X: np.ndarray
        :returns: two dimensional feature array
        :rtype: np.ndarray
        """
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for branch in self._branches:
            k = branch.nfeatures()
            result[:, index:index+k] = branch.fit_transform(X)
            index += k
        return result

    def clear_cache(self):
        """Executes the clear_cache method on all branches in the Fruit
        object.
        """
        for branch in self._branches:
            branch.clear_cache()

    def copy(self):
        """Creates a copy of this Fruit object.
        This also creates copies of all branches in this object.
        
        :returns: Copy of this Fruit object
        :rtype: Fruit
        """
        copy_ = Fruit(self.name+" (Copy)")
        copy_._branches = [self._current_branch.copy()]
        copy_._current_branch = copy_._branches[0]
        for i in range(len(self._branches)):
            if i != self._current_branch_index:
                copy_.add_branch(self._branches[i])
        return copy_

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
    The number of features for one time series is equal to:
        [number of iterators] x [number of sieves] 
    """
    def __init__(self):
        # lists of used classes for data processing
        self._preparateurs = []
        self._iterators = []
        self._sieves = []

        # list with inner lists containing sieves
        # all sieves in one list are trained on one specific output
        # of an ISS-result
        self._sieves_extended = []

        # input data is specified by calling FruitBranch.fit
        self._input_data = None

        # data variables that hold the processed data
        self._prepared_data = None
        self._iterated_data = None
        self._sieved_data = None

    def add_preparateur(self, preparateur: DataPreparateur):
        """Adds a DataPreparateur object to the branch.
        
        :param preparateur: New preparateur
        :type preparateur: DataPreparateur
        """
        if not isinstance(preparateur, DataPreparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._prepared_data = None
        self._iterated_data = None

    def get_preparateurs(self) -> list:
        """Returns a list of all DataPreparateur objects added to the
        branch.
        
        :returns: List of DataPreparateur objects
        :rtype: list
        """
        return self._preparateurs

    def clear_preparateurs(self):
        """Removes all DataPreparateur objects that were added to this
        branch.
        """
        self._preparateurs = []
        self._prepared_data = None
        self._iterated_data = None

    def add_iterator(self, sum_iter: SummationIterator):
        """Adds a SummationIterator object to the branch.
        
        :param preparateur: New iterator
        :type preparateur: SummationIterator
        """
        if not isinstance(sum_iter, SummationIterator):
            raise TypeError
        self._iterators.append(sum_iter)
        self._iterated_data = None

    def get_iterators(self) -> list:
        """Returns a list of all SummationIterator objects added to the
        branch.
        
        :returns: List of SummationIterator objects
        :rtype: list
        """
        return self._iterators

    def clear_iterators(self):
        """Removes all SummationIterator objects that were added to this
        branch.
        """
        self._iterators = []
        self._iterated_data = None

    def add_sieve(self, feat: FeatureSieve):
        if not isinstance(feat, FeatureSieve):
            raise TypeError
        self._sieves.append(feat)

    def get_sieves(self) -> list:
        """Returns a list of all FeatureSieve objects added to the
        branch.
        
        :returns: List of FeatureSieve objects
        :rtype: list
        """
        return self._sieves

    def clear_sieves(self):
        """Removes all FeatureSieve objects that were added to this
        branch.
        """
        self._sieves = []

    def add(self, *objects):
        """Adds one or multiple object(s) to the branch.
        These objects can be of type:
        - fruits.preparateurs.DataPreparateur
        - fruits.iterators.SummationIterator
        - fruits.features.FeatureSieve
        
        :param objects: Object(s) to add to the branch
        :type objects: Object of mentioned type(s) or iterable object
        containing multiple objects of mentioned type(s)
        :raises: TypeError if one of the objects has an unknown type
        """
        objects = np.array(objects, dtype=object).flatten()
        for obj in objects:
            if inspect.isclass(obj):
                obj = obj()
            if isinstance(obj, DataPreparateur):
                self.add_preparateur(obj)
            elif isinstance(obj, SummationIterator):
                self.add_iterator(obj)
            elif isinstance(obj, FeatureSieve):
                self.add_sieve(obj)
            else:
                raise TypeError("Cannot add variable of unknown type "+
                                str(type(obj)))

    def clear(self):
        """Clears all settings, configurations and calculated results
        the branch has.
        
        After the branch is cleared, it has the same settinga as a newly
        created FruitBranch object.
        """
        self.clear_preparateurs()
        self.clear_iterators()
        self.clear_sieves()
        self._sieves_extended = []

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.
        
        :returns: number of features
        :rtype: int
        """
        return len(self._sieves) * len(self._iterators)

    def fit(self, X: np.ndarray):
        """Fits the branch to the given dataset. What this action
        explicitly does depends on the added preparateurs, iterators
        and sieves.
        
        :param X: (multidimensional) time series dataset;
        If `X.ndims < 3` then the array will be expanded to contain 3
        dimensions. This could lead to unwanted behaviour.
        :type X: np.ndarray
        :raises: ValueError if `X.ndims > 3`
        """
        self._sieves_extended = []
        self._input_data = self._reshape_input(X)
        self._prepared_data = self._prepare(self._input_data)
        self._iterated_data = self._iterate(self._prepared_data)
        for i in range(len(self._iterators)):
            sieves_copy = [x.copy() for x in self._sieves]
            for sieve in sieves_copy:
                sieve.fit(self._iterated_data[:, i, :])
            self._sieves_extended.append(sieves_copy)

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        if out.ndim == 1:
            out = np.expand_dims(out, axis=0)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=1)
        if out.ndim != 3:
            raise ValueError("Unsupported input shape")
        return out

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        prepared_data = X.copy()
        for prep in self._preparateurs:
            prepared_data = prep(prepared_data)
        return prepared_data

    def _iterate(self, X: np.ndarray) -> np.ndarray:
        if not self._iterators:
            raise RuntimeError("No SummationIterator specified")
        if self._prepared_data is None:
            raise RuntimeError("Data hasn't been preparated yet")
        iterated_data = np.zeros((X.shape[0], len(self._iterators), 
                                   X.shape[2]))
        iterated_data = ISS(X.copy(), self._iterators)
        return iterated_data

    def _sieve(self, X: np.ndarray) -> np.ndarray:
        if not self._sieves:
            raise RuntimeError("No FeatureSieve specified")
        if self._iterated_data is None:
            raise RuntimeError("Iterated sums aren't calculated yet")
        sieved_data = np.zeros((X.shape[0], self.nfeatures()))
        k = 0
        X_copy = X.copy()
        for i in range(len(self._iterators)):
            for sieve in self._sieves_extended[i]:
                sieved_data[:, k] = sieve.sieve(X_copy[:, i, :])
                k += 1
        return sieved_data

    def transform(self, X: np.ndarray = None) -> np.ndarray:
        """Transforms the given time series dataset. The results are
        the calculated features for the different time series.
        
        :param X: (multidimensional) time series dataset
        If nothing is supplied, X will be set to the dataset specified
        in the last FruitBranch.fit call., defaults to None
        :type X: np.ndarray, optional
        :returns: Features for all time series in X
        :rtype: np.ndarray
        :raises: RuntimeError if FruitBranch.fit wasn't called
        """
        if self._input_data is None:
            raise RuntimeError("Missing call of FruitBranch.fit")
        if X is not None:
            reshaped_X = self._reshape_input(X)
            if not np.array_equal(self._input_data, reshaped_X):
                self._input_data = reshaped_X
                self._prepared_data = self._prepare(self._input_data)
                self._iterated_data = self._iterate(self._prepared_data)
        return self._sieve(self._iterated_data)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """This function does the same that calling `FruitBranch.fit(X)`
        and `FruitBranch.transform(X)` consecutively does.
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        :returns: transformed time series (features)
        :rtype: np.ndarray
        """
        self.fit(X)
        return self.transform(X)

    def clear_cache(self):
        """Clears all cached data but doesn't change or remove the added
        DataPreparateur/SummationIterator/FeatureSieve objects.
        """
        self._input_data = None
        self._prepared_data = None
        self._iterated_data = None
        self._sieves_extended = []

    def copy(self):
        """Returns a copy of this FruitBranch object.
        
        :returns: Copy of the branch with same settings but all
        calculations done erased.
        :rtype: FruitBranch
        """
        copy_ = FruitBranch()
        for preparateur in self.get_preparateurs():
            copy_.add(preparateur)
        for iterator in self.get_iterators():
            copy_.add(iterator)
        for sieve in self.get_sieves():
            copy_.add(sieve)
        return copy_

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def __copy__(self):
        return self.copy()
