import re
import inspect
from typing import List, Union

import numpy as np

from fruits.requisites import RequisiteContainer
from fruits.base.scope import force_input_shape
from fruits.base.callback import AbstractCallback
from fruits.core.iss import ISSCalculator
from fruits.core.wording import Word
from fruits.sieving.abstract import FeatureSieve
from fruits.preparation.abstract import DataPreparateur

class Fruit:
    """Feature Extractor using iterated sums.

    A Fruit object consists of a number of ``FruitBranch`` objects.
    At the end of the pipeline, each branch returns their own features
    and they will be concatenated by this class.

    A simple example (using one ``FruitBranch``):

    .. code-block:: python

        fruit = fruits.Fruit("My Fruit")
        # optional: add preparateurs for preprocessing
        fruit.add(fruits.preparation.INC(zero_padding=False))
        # add words for iterated sums calculation
        fruit.add(fruits.core.generation.simplewords_by_degree(2, 2, 1))
        # choose sieves
        fruit.add(fruits.sieving.PPV(0.5))
        fruit.add(fruits.sieving.MAX)
        fruit.add(fruits.sieving.MIN)
        fruit.add(fruits.sieving.END)

        # transform time series dataset
        fruit.fit(X_train)
        X_train_transformed = fruit.transform(X_train)
        X_test_tranformed = fruit.transform(X_test)

        # use the transformed results (features) in a classifier
        ...

    The above defined ``fruit`` will result in ``6*4=24`` features per
    time series.

    Calling ``add(...)`` on a ``Fruit`` object always calls
    ``branch.add(...)`` where ``branch`` is the currently selected
    ``FruitBranch`` in this object. What branch is selected can be
    changed by calling ``self.switch_branch(index)`` or forking a new
    branch with ``self.fork()``.
    """
    def __init__(self, name: str = ""):
        self.name = name
        # list of FruitBranches
        self._branches = []
        # pointer for the current branch index
        self._cbi = None
        self._fitted = False

    @property
    def name(self) -> str:
        """Simple identifier for the Fruit object."""
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name

    def fork(self, branch: "FruitBranch" = None):
        """Adds a new branch to the pipeline. If none is given, an
        empty FruitBranch will be created and switched to.

        :type branch: FruitBranch, optional
        """
        if branch is None:
            branch = FruitBranch()
        self._branches.append(branch)
        self._cbi = len(self._branches) - 1
        self._fitted = False

    def branch(self, index: int = None):
        """Returns the currently selected branch or the branch with the
        given index.

        :rtype: FruitBranch
        """
        if index is None:
            return self._branches[self._cbi]
        return self._branches[index]

    def branches(self) -> list:
        """Returns all branches of this Fruit object.

        :rtype: list
        """
        return self._branches

    def switch_branch(self, index: int):
        """Switches to the branch with the given index.

        :param index: Integer in ``[0, 1, ..., len(self.branches())-1]``
        :type index: int
        """
        if not (0 <= index < len(self._branches)):
            raise IndexError("Index has to be in [0, len(self.branches()))")
        self._cbi = index

    def add(self, *objects: Union[DataPreparateur, Word, FeatureSieve]):
        """Adds one or multiple object(s) to the `currently selected`
        branch.
        These objects can be one of the following types:

        - :class:`~fruits.preparation.abstract.DataPreparateur`
        - :class:`~fruits.core.wording.Word`
        - :class:`~fruits.sieving.abstract.FeatureSieve`

        :type objects: Object of mentioned type(s) or iterable object
            containing multiple objects of mentioned type(s).
        :raises: TypeError if one of the objects has an unknown type
        """
        if len(self._branches) == 0:
            self.fork()
        self._branches[self._cbi].add(objects)
        self._fitted = False

    def nfeatures(self) -> int:
        """Returns the total number of features of all branches 
        combined.

        :rtype: int
        """
        return sum([branch.nfeatures() for branch in self._branches])

    def fit(self, X: np.ndarray):
        """Fits all branches to the given data.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            ``fruits.base.scope.force_input_shape``.
        :type X: np.ndarray
        """
        for branch in self._branches:
            branch.fit(X)
        self._fitted = True

    def transform(self, X: np.ndarray,
                  callbacks: List[AbstractCallback] = []) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        branches this Fruit object contains.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            ``fruits.base.scope.force_input_shape``.
        :type X: np.ndarray
        :param callbacks: List of callbacks. To write your own callback,
            override the class ``fruits.callback.AbstractCallback``.,
            defaults to empty list
        :type callbacks: List[AbstractCallback], optional
        :rtype: np.ndarray
        :raises: RuntimeError if Fruit.fit wasn't called
        """
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")
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
        self.fit(X)
        return self.transform(X)

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains a
        summary for each FruitBranch in this Fruit object.

        :rtype: str
        """
        summary = "{:=^80}".format(f"Summary of fruits.Fruit: '{self.name}'")
        summary += f"\nBranches: {len(self.branches())}"
        summary += f"\nFeatures: {self.nfeatures()}"
        for branch in self.branches():
            summary += "\n\n" + branch.summary()
        summary += "\n{:=^80}".format(f"End of Summary")
        return summary

    def copy(self) -> "Fruit":
        """Creates a shallow copy of this Fruit object.
        This also creates shallow copies of all branches in this object.

        :rtype: Fruit
        """
        copy_ = Fruit(self.name+" (Copy)")
        for branch in self._branches:
            copy_.fork(branch.copy())
        return copy_

    def deepcopy(self) -> "Fruit":
        """Creates a deep copy of this Fruit object.
        This also creates deep copies of all branches in this object.

        :rtype: Fruit
        """
        copy_ = Fruit(self.name+" (Copy)")
        for branch in self._branches:
            copy_.fork(branch.deepcopy())
        return copy_

    def __copy__(self) -> "Fruit":
        return self.copy()

    def __deepcopy__(self) -> "Fruit":
        return self.deepcopy()


class FruitBranch:
    """One branch for a Fruit object.

    A FruitBranch object extracts values from time series data that are 
    somehow representative of the input data.
    The user can customize any of the following three steps the 
    extractor is going to do in order to get the so called features.

    Preparing data:
    Apply functions at the start of the extraction procedure.
    There are many so called
    :class:`~fruits.preparation.abstract.DataPreparateur` objects in
    fruits available for preprocessing.
    The preparateurs will be applied sequentially to the input data.

    Calculating Iterated Sums:
    The preprocessed data is now used to calculate the iterated sums
    signature for different :class:`~fruits.core.wording.Word` objects
    the user can specify.

    Extracting Features:
    Each :class:`~fruits.sieving.abstract.FeatureSieve` added to the
    branch will be fitted on the iterated sums from the previous step.
    The branch then returns an array of numbers (the transformed results
    from those sieves), i.e. the features for each time series.
    """
    def __init__(self):
        # lists of used classes for data processing
        self._preparateurs = []
        self._words = []
        self._sieves = []

        # calculator used for the ISS calculation
        self._calculator = ISSCalculator()
        self._calculator.batch_size = 1

        # list with inner lists containing sieves
        # all sieves in one list are trained on one specific output
        # of an ISS-result
        self._sieves_extended = []

        # bool variable that is True if the FruitBranch is fitted
        self._fitted = False

        # list of calculations that are shared among sieves
        self._sieve_prerequisites = None

    @property
    def calculator(self):
        """Returns the used calculator for the iterated sums. This
        for example allows setting the mode used in the calculation.

        :rtype: ISSCalculator
        """
        return self._calculator

    def add_preparateur(self, preparateur: DataPreparateur):
        """Adds a preparateur to the branch.

        :type preparateur: DataPreparateur
        """
        if not isinstance(preparateur, DataPreparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._fitted = False

    def get_preparateurs(self) -> List[DataPreparateur]:
        """Returns a list of all preparateurs added to the
        branch.

        :rtype: List[DataPreparateur]
        """
        return self._preparateurs

    def clear_preparateurs(self):
        """Removes all preparateurs that were added to this branch."""
        self._preparateurs = []
        self._fitted = False

    def add_word(self, word: Word):
        """Adds a word to the branch.

        :type word: Word
        """
        if not isinstance(word, Word):
            raise TypeError
        self._words.append(word)
        self._fitted = False

    def get_words(self) -> List[Word]:
        """Returns a list of all words in the branch.

        :rtype: List[Word]
        """
        return self._words

    def clear_words(self):
        """Removes all words that were added to this branch."""
        self._words = []
        self._sieves_extended = []
        self._fitted = False

    def add_sieve(self, sieve: FeatureSieve):
        """Appends a new feature sieve to the FruitBranch.

        :type sieve: FeatureSieve
        """
        if not isinstance(sieve, FeatureSieve):
            raise TypeError
        self._sieves.append(sieve)
        self._fitted = False

    def get_sieves(self) -> List[FeatureSieve]:
        """Returns a list of all feature sieves added to the branch.

        :rtype: List[FeatureSieve]
        """
        return self._sieves

    def clear_sieves(self):
        """Removes all feature sieves that were added to this branch.
        """
        self._sieves = []
        self._sieve_prerequisites = None
        self._sieves_extended = []
        self._fitted = False

    def add(self, *objects: Union[DataPreparateur, Word, FeatureSieve]):
        """Adds one or multiple object(s) to the branch.
        These objects can be of type:

        - :class:`~fruits.preparation.abstract.DataPreparateur`
        - :class:`~fruits.core.wording.Word`
        - :class:`~fruits.sieving.abstract.FeatureSieve`

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
            elif isinstance(obj, Word):
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
        return (sum([s.nfeatures() for s in self._sieves])
                * self.calculator._n_iterated_sums(self._words))

    def _compile(self):
        # checks if the FruitBranch is configured correctly and ready
        # for fitting
        if not self._words:
            raise RuntimeError("No words specified for ISS calculation")
        if not self._sieves:
            raise RuntimeError("No FeatureSieve objects specified")

    def _collect_requisites(self):
        # collects requisites of all added preparateurs and sieves
        for prep in self._preparateurs:
            if (req := prep._requisite) is not None:
                self._requisite_container.register(req)
                prep._set_requisite_container(self._requisite_container)
        for sieve in self._sieves:
            if (req := sieve._requisite) is not None:
                self._requisite_container.register(req)
                sieve._set_requisite_container(self._requisite_container)

    def _select_fit_sample(self, X: np.ndarray) -> np.ndarray:
        # returns a sample of the data used for fitting
        ind = np.random.randint(0, X.shape[0])
        return X[ind:ind+1, :, :]

    def fit(self, X: np.ndarray):
        """Fits the branch to the given dataset. What this action
        explicitly does depends on the FruitBranch configuration.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            :meth:`fruits.base.scope.force_input_shape`.
        :type X: np.ndarray
        :raises: ValueError if ``X.ndims > 3``
        """
        self._compile()

        self._requisite_container = RequisiteContainer()
        self._collect_requisites()
        self._requisite_container.process(force_input_shape(X))
        prepared_data = self._select_fit_sample(X)
        for prep in self._preparateurs:
            prep.fit(prepared_data)
            prepared_data = prep.prepare(prepared_data)

        self._sieves_extended = []
        self.calculator.start(prepared_data, self._words)
        for iterated_data in self.calculator:
            iterated_data = iterated_data.reshape(iterated_data.shape[0]
                                                  * iterated_data.shape[1],
                                                  iterated_data.shape[2])
            sieves_copy = [sieve.copy() for sieve in self._sieves]
            for sieve in sieves_copy:
                sieve.fit(iterated_data[:, :])
            self._sieves_extended.append(sieves_copy)
        self._fitted = True

    def transform(self, X: np.ndarray,
                  callbacks: List[AbstractCallback] = []) -> np.ndarray:
        """Transforms the given time series dataset. The results are
        the calculated features for the different time series.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            ``fruits.base.scope.force_input_shape``.
        :type X: np.ndarray
        :param callbacks: List of callbacks. To write your own callback,
            override the class ``fruits.callback.AbstractCallback``.,
            defaults to empty list
        :type callbacks: List[AbstractCallback], optional
        :rtype: np.ndarray
        :raises: ValueError if ``X.ndims > 3``
        :raises: RuntimeError if FruitBranch.fit wasn't called
        """
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")

        self._requisite_container.clear()
        self._requisite_container.process(X)
        prepared_data = force_input_shape(X)
        for prep in self._preparateurs:
            prepared_data = prep.prepare(prepared_data)
            for callback in callbacks:
                callback.on_preparateur(prepared_data)
        for callback in callbacks:
            callback.on_preparation_end(prepared_data)

        sieved_data = np.zeros((prepared_data.shape[0],
                                self.nfeatures()))
        k = 0
        self.calculator.start(prepared_data, self._words)
        for i, iterated_data in enumerate(self.calculator):
            for callback in callbacks:
                callback.on_iterated_sum(iterated_data)
            for j, sieve in enumerate(self._sieves_extended[i]):
                nf = sieve.nfeatures()
                new_features = nf * iterated_data.shape[1]
                for it in range(iterated_data.shape[1]):
                    sieved_data[:, k+it*nf:k+(it+1)*nf] = \
                            sieve.sieve(iterated_data[:, it, :])
                for callback in callbacks:
                    callback.on_sieve(sieved_data[k:k+new_features])
                k += new_features
        for callback in callbacks:
            callback.on_sieving_end(sieved_data)
        return sieved_data

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """This function does the same that calling ``self.fit(X)`` and
        ``self.transform(X)`` consecutively does.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            ``fruits.base.scope.force_input_shape``.
        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        self.fit(X)
        return self.transform(X)

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains all
        added preparateurs, words and sieves.

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
        elif len(self._words) > 10:
            summary += "\n\t+ " + \
                       "\n\t+ ".join([str(x) for x in self._words[:9]])
            summary += "\n\t..."
        else:
            summary += "\n\t+ " + \
                       "\n\t+ ".join([str(x) for x in self._words])
        summary += f"\nSieves ({len(self._sieves)}): "
        if len(self._sieves) == 0:
            summary += "-"
        else:
            for x in self._sieves:
                lines = x.summary().split("\n")
                summary += "\n\t+ " + lines[0]
                summary += "\n\t  "
                summary += "\n\t  ".join(lines[1:])
        return summary

    def copy(self) -> "FruitBranch":
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

    def deepcopy(self) -> "FruitBranch":
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
        copy_._calculator = self.calculator.copy()
        return copy_

    def __copy__(self) -> "FruitBranch":
        return self.copy()

    def __deepcopy__(self) -> "FruitBranch":
        return self.deepcopy()
