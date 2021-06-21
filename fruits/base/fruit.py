import re
import inspect

import numpy as np

from fruits.base.callback import AbstractCallback
from fruits.preparation import DataPreparateur
from fruits.core.wording import AbstractWord
from fruits.sieving import FeatureSieve
from fruits.base import scope
from fruits import core

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
        """Simple identifier for the Fruit object."""
        return self._name
    
    @name.setter
    def name(self, name:str):
        self._name = name

    def add_branch(self, branch):
        """Adds a new branch to the pipeline.

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
        
        :rtype: list
        """
        return self._branches

    def switch_branch(self, index: int):
        """Switches to the branch with the given index. If the Fruit
        objects points to a branch, then each method that is called on
        the Fruit object is actually called on this branch.
        
        :param index: Integer in ``[0, 1, ..., len(self.branches)-1]``
        :type index: int
        """
        self._current_branch = self._branches[index]
        self._current_branch_index = index

    def current_branch(self):
        """Returns the branch that is currently selected.

        The selected branch is used to extend the pipeline by calling
        functions like `add()` on the Fruit object. Normally this is
        done on a FruitBranch object. 
        
        :rtype: FruitBranch
        """
        return self._current_branch

    def current_branch_index(self) -> int:
        """Returns the index of the currently selected branch.
        
        :rtype: int
        """
        return self._current_branch_index

    def add_preparateur(self, preparateur: DataPreparateur):
        """Adds a DataPreparateur object to the currently selected
        branch.

        :type preparateur: DataPreparateur
        """
        self._current_branch.add_preparateur(preparateur)

    def get_preparateurs(self) -> list:
        """Returns all DataPreparateur objects that are added to the
        currently selected branch.
        
        :rtype: list
        """
        return self._current_branch.get_preparateurs()

    def clear_preparateurs(self):
        """Removes all added DataPreparateur objects in the currently
        selected branch.
        """
        self._current_branch.clear_preparateurs()

    def add_word(self, sum_iter: AbstractWord):
        """Adds a AbstractWord object to the currently selected
        branch.

        :param preparateur: New AbstractWord
        :type preparateur: AbstractWord
        """
        self._current_branch.add_word(sum_iter)

    def get_words(self) -> list:
        """Returns all AbstractWord objects that are added to the
        currently selected branch.
        
        :rtype: list
        """
        return self._current_branch.get_words()

    def clear_iterators(self):
        """Removes all added AbstractWord objects in the currently
        selected branch.
        """
        self._current_branch.clear_iterators()
        
    def add_sieve(self, feat: FeatureSieve):
        """Adds a FeatureSieve object to the currently selected branch.

        :type preparateur: FeatureSieve
        """
        self._current_branch.add_feature_sieve(feat)

    def get_sieves(self) -> list:
        """Returns all FeatureSieve objects that are added to the
        currently selected branch.
        
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
        - classes that inherit from fruits.core.wording.AbstractWord
        - fruits.features.FeatureSieve
        
        :type objects: Object of mentioned type(s) or iterable object
            containing multiple objects of mentioned type(s).
        :raises: TypeError if one of the objects has an unknown type
        """
        self._current_branch.add(objects)

    def clear(self):
        """Clears all DataPreparateur, AbstractWord and
        FeatureSieve objects in the currently selected branch.
        """
        self._current_branch.clear()

    def nfeatures(self) -> int:
        """Returns the total number of features of all branches 
        combined.
        
        :rtype: int
        """
        return sum([branch.nfeatures() for branch in self._branches])

    def fit(self, X: np.ndarray):
        """Fits all branches to the given data.
        
        :param X: (Multidimensional) time series dataset
        :type X: np.ndarray
        """
        for branch in self._branches:
            branch.fit(X)

    def transform(self, X: np.ndarray, callbacks: list = []) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        branches this Fruit object contains.
        
        :param X: (Multidimensional) time series dataset
        :type X: np.ndarray
        :param callbacks: List of callbacks. To write your own callback,
            override the class ``fruits.callback.AbstractCallback``.,
            defaults to empty list
        :type callbacks: list of AbstractCallback objects, optional
        :rtype: np.ndarray
        :raises: RuntimeError if Fruit.fit wasn't called
        """
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for branch in self._branches:
            for callback in callbacks:
                callback.on_next_branch()
            k = branch.nfeatures()
            result[:, index:index+k] = branch.transform(X, callbacks)
            index += k
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits all branches to the given dataset and returns the
        transformed results of X from all branches.
        
        :param X: (Multidimensional) time series dataset
        :type X: np.ndarray
        :returns: Two dimensional feature array
        :rtype: np.ndarray
        """
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for branch in self._branches:
            k = branch.nfeatures()
            result[:, index:index+k] = branch.fit_transform(X)
            index += k
        return result

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains a
        summary for each FruitBranch in this Fruit object.
        
        :rtype: str
        """
        summary = "{:=^80}".format(f"Summary of fruits.Fruit: '{self.name}'")
        summary += f"\nBranches: {len(self.branches())}"
        summary += f"\nFeatures: {self.nfeatures()}"
        for branch in self.branches():
            summary += "\n\n"+branch.summary()
        summary += "\n{:=^80}".format(f"End of Summary")
        return summary

    def copy(self):
        """Creates a shallow copy of this Fruit object.
        This also creates shallow copies of all branches in this object.
        
        :rtype: Fruit
        """
        copy_ = Fruit(self.name+" (Copy)")
        copy_._branches = [self._current_branch.copy()]
        copy_._current_branch = copy_._branches[0]
        for i in range(len(self._branches)):
            if i != self._current_branch_index:
                copy_.add_branch(self._branches[i].copy())
        return copy_

    def deepcopy(self):
        """Creates a deep copy of this Fruit object.
        This also creates deep copies of all branches in this object.
        
        :rtype: Fruit
        """
        copy_ = Fruit(self.name+" (Copy)")
        copy_._branches = [self._current_branch.deepcopy()]
        copy_._current_branch = copy_._branches[0]
        for i in range(len(self._branches)):
            if i != self._current_branch_index:
                copy_.add_branch(self._branches[i].deepcopy())
        return copy_

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.deepcopy()


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
    signature for different words the user can specify.

    Extracting the Features:
    Each FeatureSieve added to the branch will be fitted and on 
    the iterated sums from the previous step. The Fruit object then  
    returns an array of numbers (the transformed results from those
    sieves), i.e. the features for each time series.
    """
    def __init__(self):
        # lists of used classes for data processing
        self._preparateurs = []
        self._words = []
        self._sieves = []

        # list with inner lists containing sieves
        # all sieves in one list are trained on one specific output
        # of an ISS-result
        self._sieves_extended = []

        # bool variable that is True if the FruitBranch is fitted
        self._fitted = False

        # list of calculations that are shared among sieves
        self._sieve_prerequisites = None

    def add_preparateur(self, preparateur: DataPreparateur):
        """Adds a DataPreparateur object to the branch.
        
        :param preparateur: New preparateur
        :type preparateur: DataPreparateur
        """
        if not isinstance(preparateur, DataPreparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._fitted = False

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
        self._fitted = False

    def add_word(self, word: AbstractWord):
        """Adds a word to the branch.
        
        :param preparateur: New ComplexWord or SimpleWord.
        :type preparateur: AbstractWord
        """
        if not isinstance(word, AbstractWord):
            raise TypeError
        self._words.append(word)
        self._fitted = False

    def get_words(self) -> list:
        """Returns a list of all words in the branch.
        
        :rtype: list
        """
        return self._words

    def clear_words(self):
        """Removes all words that were added to this branch."""
        self._words = []
        self._sieves_extended = []
        self._fitted = False

    def add_sieve(self, feat: FeatureSieve):
        """Appends a new FeatureSieve to the FruitBranch.
        
        :type feat: FeatureSieve
        """
        if not isinstance(feat, FeatureSieve):
            raise TypeError
        self._sieves.append(feat)
        self._fitted = False

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
        self._sieve_prerequisites = None
        self._sieves_extended = []
        self._fitted = False

    def add(self, *objects):
        """Adds one or multiple object(s) to the branch.
        These objects can be of type:
        
        - fruits.preparateurs.DataPreparateur
        - classes that inherit from fruits.core.wording.AbstractWord
        - fruits.features.FeatureSieve
        
        :type objects: Object(s) of mentioned type(s) or iterable object
            containing multiple objects of mentioned type(s).
        :raises: TypeError if one of the objects has an unknown type
        """
        objects = np.array(objects, dtype=object).flatten()
        for obj in objects:
            if inspect.isclass(obj):
                obj = obj()
            if isinstance(obj, DataPreparateur):
                self.add_preparateur(obj)
            elif isinstance(obj, AbstractWord):
                self.add_word(obj)
            elif isinstance(obj, FeatureSieve):
                self.add_sieve(obj)
            else:
                raise TypeError("Cannot add variable of type"+str(type(obj)))

    def clear(self):
        """Clears all settings, configurations and calculated results
        the branch has.
        
        After the branch is cleared, it has the same settings as a newly
        created FruitBranch object.
        """
        self.clear_preparateurs()
        self.clear_words()
        self.clear_sieves()

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.
        
        :rtype: int
        """
        return sum([s.nfeatures() for s in self._sieves])*len(self._words)

    def _compile(self):
        # checks if the FruitBranch is configured correctly and ready
        # for fitting
        if not self._words:
            raise RuntimeError("No words specified for ISS calculation")
        if not self._sieves:
            raise RuntimeError("No FeatureSieve objects specified")

    def fit(self, X: np.ndarray):
        """Fits the branch to the given dataset. What this action
        explicitly does depends on the FruitBranch configuration.
        
        :param X: Multidimensional time series dataset.
            If ``X.ndims < 3`` then the array will be expanded to
            contain 3 dimensions. This could lead to unwanted behaviour.
        :type X: np.ndarray
        :raises: ValueError if ``X.ndims > 3``
        """
        self._compile()

        prepared_data = scope.force_input_shape(X)
        for prep in self._preparateurs:
            prepared_data = prep.fit_prepare(prepared_data)

        self._sieves_extended = []
        for i in range(len(self._words)):
            iterated_data = core.iss.ISS(prepared_data, self._words[i])
            sieves_copy = [sieve.copy() for sieve in self._sieves]
            for sieve in sieves_copy:
                sieve.fit(iterated_data[:, 0, :])
            self._sieves_extended.append(sieves_copy)
        self._fitted = True

    def transform(self, X: np.ndarray, callbacks: list = []) -> np.ndarray:
        """Transforms the given time series dataset. The results are
        the calculated features for the different time series.
        
        :param X: Multidimensional time series dataset.
            If ``X.ndims < 3`` then the array will be expanded to
            contain 3 dimensions. This could lead to unwanted behaviour.
        :type X: np.ndarray
        :param callbacks: List of callbacks. To write your own callback,
            override the class ``fruits.callback.AbstractCallback``.,
            defaults to empty list
        :type callbacks: list of AbstractCallback objects, optional
        :rtype: np.ndarray
        :raises: ValueError if ``X.ndims > 3``
        :raises: RuntimeError if FruitBranch.fit wasn't called
        """
        if not self._fitted:
            raise RuntimeError("Missing call of FruitBranch.fit")

        input_data = scope.force_input_shape(X)

        # calculates prerequisites for feature sieves
        _sieve_prerequisites = []
        for i, sieve in enumerate(self._sieves):
            prereq = sieve._prerequisites()
            for j in range(i):
                if prereq == _sieve_prerequisites[j]:
                    _sieve_prerequisites.append(_sieve_prerequisites[j])
                    break
            else:
                _sieve_prerequisites.append(prereq)
        for i in range(len(self._sieves)):
            if not _sieve_prerequisites[i].is_processed():
                _sieve_prerequisites[i].process(input_data)

        prepared_data = input_data
        for prep in self._preparateurs:
            prepared_data = prep.prepare(prepared_data)
            for callback in callbacks:
                callback.on_preparateur(prepared_data)
        for callback in callbacks:
            callback.on_preparation_end(prepared_data)

        sieved_data = np.zeros((prepared_data.shape[0],
                                self.nfeatures()))
        k = 0
        for i in range(len(self._words)):
            iterated_data = core.iss.ISS(prepared_data, self._words[i])
            for callback in callbacks:
                callback.on_iterated_sum(iterated_data)
            for j, sieve in enumerate(self._sieves_extended[i]):
                sieve._load_prerequisites(_sieve_prerequisites[j])
                new_features = sieve.nfeatures()
                if new_features == 1:
                    sieved_data[:, k] = sieve.sieve(iterated_data[:, 0, :])
                else:
                    sieved_data[:, k:k+new_features] = \
                            sieve.sieve(iterated_data[:, 0, :])
                for callback in callbacks:
                    callback.on_sieve(sieved_data[k:k+new_features])
                k += new_features
        for callback in callbacks:
            callback.on_sieving_end(sieved_data)
        return sieved_data

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

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains all
        added DataPreparateur, AbstractWord and FeatureSieve
        objects.
        
        :returns: Summary as string
        :rtype: str
        """
        summary = "{:-^80}".format("fruits.FruitBranch")
        summary += f"\nNumber of features: {self.nfeatures()}"
        summary += f"\n\nPreparateurs ({len(self._preparateurs)}): "
        if len(self._preparateurs) == 0:
            summary += "-" 
        else: 
            summary += "\n\t+ " + \
                       "\n\t+ ".join([str(x) for x in self._preparateurs])
        summary += f"\nIterators ({len(self._words)}): "
        if len(self._words) == 0:
            summary += "-" 
        else: 
            summary += "\n\t+ " + \
                       "\n\t+ ".join([str(x) for x in self._words])
        summary += f"\nSieves ({len(self._sieves)}): "
        if len(self._sieves) == 0:
            summary += "-" 
        else: 
            summary += "\n\t+ " + \
                       "\n\t+ ".join([str(x) for x in self._sieves])
        return summary

    def copy(self):
        """Returns a shallow copy of this FruitBranch object.
        
        :returns: Copy of the branch with same settings but all
            calculations done erased.
        :rtype: FruitBranch
        """
        copy_ = FruitBranch()
        for preparateur in self._preparateurs:
            copy_.add(preparateur)
        for iterator in self._words:
            copy_.add(iterator)
        for sieve in self._sieves:
            copy_.add(sieve)
        return copy_

    def deepcopy(self):
        """Returns a deep copy of this FruitBranch object.
        
        :returns: Deepcopy of the branch with same settings but all
            calculations done erased.
        :rtype: FruitBranch
        """
        copy_ = FruitBranch()
        for preparateur in self._preparateurs:
            copy_.add(preparateur.copy())
        for iterator in self._words:
            copy_.add(iterator.copy())
        for sieve in self._sieves:
            copy_.add(sieve.copy())
        return copy_

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.deepcopy()
