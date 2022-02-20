import inspect
from typing import Optional, Union, Any

import numpy as np

from fruits.cache import Cache, CoquantileCache
from fruits.scope import force_input_shape, FitTransform
from fruits.core.callback import AbstractCallback
from fruits.signature.iss import SignatureCalculator, CachePlan
from fruits.words.word import Word
from fruits.sieving.abstract import FeatureSieve
from fruits.preparation.abstract import DataPreparateur


class Fruit:
    """Feature Extractor using iterated sums.

    A Fruit consists of a number of
    :class:`~fruits.core.fruit.FruitBranch` objects.
    At the end of the pipeline, each branch returns their own features
    and they will be concatenated by this class.

    A simple example (using two branches):

    .. code-block:: python

        fruit = fruits.Fruit("My Fruit")
        # optional: add preparateurs for preprocessing
        fruit.add(fruits.preparation.INC)
        # add words for iterated sums calculation
        fruit.add(fruits.words.creation.simplewords_by_weight(4))
        # choose sieves
        fruit.add(fruits.sieving.PPV(0.5))
        fruit.add(fruits.sieving.END)

        # add a new branch without INC
        fruit.fork()
        fruit.add(fruits.words.creation.simplewords_by_weight(4))
        fruit.add(fruits.sieving.PPV(0.5))
        fruit.add(fruits.sieving.END)

        # configure the fruit
        fruit.configure(mode="extended")

        # fit the fruit on a time series dataset
        fruit.fit(X_train)
        # transform the dataset
        X_train_transformed = fruit.transform(X_train)
        X_test_tranformed = fruit.transform(X_test)

        # use the transformed results (features) in a classifier
        ...

    The ``fruit`` above will result in ``2*8*2=32`` features per time
    series.
    """

    def __init__(self, name: str = ""):
        self.name: str = name
        # list of FruitBranches
        self._branches: list[FruitBranch] = []
        # pointer for the current branch index
        self._cbi: int = 0
        self._fitted: bool = False

    def fork(self, branch: "FruitBranch" = None) -> None:
        """Adds a new branch to the pipeline. If none is given, an
        empty FruitBranch will be created and switched to.

        :type branch: FruitBranch, optional
        """
        if branch is None:
            branch = FruitBranch()
        self._branches.append(branch)
        self._cbi = len(self._branches) - 1
        self._fitted = False

    def branch(self, index: Optional[int] = None) -> "FruitBranch":
        """Returns the currently selected branch or the branch with the
        given index.

        :rtype: FruitBranch
        """
        if index is None:
            return self._branches[self._cbi]
        return self._branches[index]

    def branches(self) -> list["FruitBranch"]:
        """Returns all branches of this Fruit object.

        :rtype: list
        """
        return self._branches

    def switch_branch(self, index: int) -> None:
        """Switches to the branch with the given index.

        :param index: Integer in ``[0, 1, ..., len(self.branches())-1]``
        :type index: int
        """
        if not (0 <= index < len(self._branches)):
            raise IndexError("Index has to be in [0, len(self.branches()))")
        self._cbi = index

    def add(self, *objects: Union[FitTransform, Word, type]) -> None:
        """Adds one or multiple object(s) to the currently selected
        branch.

        :param objects: One or more objects of the following types:

            - :class:`~fruits.preparation.abstract.DataPreparateur`
            - :class:`~fruits.words.word.Word`
            - :class:`~fruits.sieving.abstract.FeatureSieve`

        :type objects: Union[FitTransform, Word]
        """
        if len(self._branches) == 0:
            self.fork()
        self._branches[self._cbi].add(*objects)
        self._fitted = False

    def nfeatures(self) -> int:
        """Returns the total number of features of all branches
        combined.

        :rtype: int
        """
        return sum([branch.nfeatures() for branch in self._branches])

    def configure(self, **kwargs: Any) -> None:
        """Makes changes to the default configuration of a all branches
        if arguments differ from ``None``.

        :param kwargs: For possible options, have a look at
            :meth:`fruits.core.fruit.FruitBranch.configure`.
        :type kwargs: Any
        """
        for branch in self._branches:
            branch.configure(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        """Fits all branches to the given data.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            :meth:`~fruits.scope.force_input_shape`.
        :type X: np.ndarray
        """
        for branch in self._branches:
            branch.fit(X)
        self._fitted = True

    def transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
    ) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        branches this Fruit object contains.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            :meth:`~fruits.scope.force_input_shape`.
        :type X: np.ndarray
        :param callbacks: list of callbacks. To write your own callback,
            override the class
            :class:`~fruits.core.callback.AbstractCallback`.,
            defaults to None
        :type callbacks: list[AbstractCallback], optional
        :rtype: np.ndarray
        :raises: RuntimeError if Fruit.fit wasn't called
        """
        if callbacks is None:
            callbacks = []
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
        result = np.nan_to_num(result, copy=False, nan=0.0)
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
        summary += f"\n{'End of Summary' : =^80}"
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


class FruitBranch:
    """One branch of a Fruit object.

    A FruitBranch object extracts values from time series data that are
    somehow representative of the input data.
    The user can customize any of the following three steps.

        - Preparing data:
            Apply functions at the start of the extraction procedure.
            There are many so called
            :class:`~fruits.preparation.abstract.DataPreparateur`
            objects in fruits available for preprocessing. The
            preparateurs will be applied sequentially to the input data.

        - Calculating Iterated Sums:
            The preprocessed data is now used to calculate the iterated
            sums signature for different
            :class:`~fruits.words.word.Word` objects the user can
            specify.

        - Extracting Features:
            Each :class:`~fruits.sieving.abstract.FeatureSieve` added to
            the branch will be fitted on the iterated sums from the
            previous step. The branch then returns an array of numbers
            (the transformed results from those sieves), i.e. the
            features for each time series.
    """

    def __init__(self):
        # lists of used classes for data processing
        self._preparateurs: list[DataPreparateur] = []
        self._words: list[Word] = []
        self._sieves: list[FeatureSieve] = []

        # calculator options used in the ISS calculation
        self._calculator_options: dict[str, Union[int, str]] = {
            "batch_size": 1,
            "mode": "single",
        }

        # list with inner lists containing sieves
        # all sieves in one list are trained on one specific output
        # of an ISS-result
        self._sieves_extended: list[list[FeatureSieve]] = []

        # configurations for fitting
        self._fitted: bool = False
        self._fit_sample_size: Union[float, int] = 1

        # cache that is calculated at fitting and also used in the
        # transformation process
        self._cache: Cache

    def configure(
        self,
        mode: str = None,
        batch_size: int = None,
        fit_sample_size: Optional[Union[float, int]] = None,
    ) -> None:
        """Makes changes to the default configuration of a fruit branch
        if arguments differ from ``None``.

        :param mode: See
            :meth:`fruits.signature.iss.SignatureCalculator.transform`,
            defaults to None
        :type mode: str, optional
        :param batch_size: See
            :meth:`~ruits.signature.iss.SignatureCalculator.transform`,
            defaults to None
        :type batch_size: int, optional
        :param fit_sample_size: Size of the random time series sample
            that is used for fitting. This is represented as a float
            which will be multiplied by ``X.shape[0]`` or ``1`` for one
            random time series., defaults to 1
        :type fit_sample_size: Union[float, int]
        """
        if mode is not None:
            self._calculator_options["mode"] = mode
        if batch_size is not None:
            self._calculator_options["batch_size"] = batch_size
        if fit_sample_size is not None:
            self._fit_sample_size = fit_sample_size

    def add_preparateur(self, preparateur: DataPreparateur) -> None:
        """Adds a preparateur to the branch.

        :type preparateur: DataPreparateur
        """
        if not isinstance(preparateur, DataPreparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._fitted = False

    def get_preparateurs(self) -> list[DataPreparateur]:
        """Returns a list of all preparateurs added to the
        branch.

        :rtype: list[DataPreparateur]
        """
        return self._preparateurs

    def clear_preparateurs(self) -> None:
        """Removes all preparateurs that were added to this branch."""
        self._preparateurs = []
        self._fitted = False

    def add_word(self, word: Word) -> None:
        """Adds a word to the branch.

        :type word: Word
        """
        if not isinstance(word, Word):
            raise TypeError
        self._words.append(word)
        self._fitted = False

    def get_words(self) -> list[Word]:
        """Returns a list of all words in the branch.

        :rtype: list[Word]
        """
        return self._words

    def clear_words(self) -> None:
        """Removes all words that were added to this branch."""
        self._words = []
        self._sieves_extended = []
        self._fitted = False

    def add_sieve(self, sieve: FeatureSieve) -> None:
        """Appends a new feature sieve to the FruitBranch.

        :type sieve: FeatureSieve
        """
        if not isinstance(sieve, FeatureSieve):
            raise TypeError
        self._sieves.append(sieve)
        self._fitted = False

    def get_sieves(self) -> list[FeatureSieve]:
        """Returns a list of all feature sieves added to the branch.

        :rtype: list[FeatureSieve]
        """
        return self._sieves

    def clear_sieves(self) -> None:
        """Removes all feature sieves that were added to this branch."""
        self._sieves = []
        self._sieves_extended = []
        self._fitted = False

    def add(self, *objects: Union[FitTransform, Word, type]) -> None:
        """Adds one or multiple object(s) to the branch.

        :type objects: One or more objects of the following types:

            - :class:`~fruits.preparation.abstract.DataPreparateur`
            - :class:`~fruits.words.word.Word`
            - :class:`~fruits.sieving.abstract.FeatureSieve`
        """
        objects_flattened = np.array(objects, dtype=object).flatten()
        for obj in objects_flattened:
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

    def clear(self) -> None:
        """Clears all settings, configurations and calculated results
        the branch has.

        After the branch is cleared, it has the same settings as a newly
        created FruitBranch object.
        """
        self.clear_preparateurs()
        self.clear_words()
        self.clear_sieves()
        self._calculator_options = {"batch_size": 1, "mode": "single"}

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.

        :rtype: int
        """
        if self._calculator_options["mode"] == "extended":
            return (
                sum([s.nfeatures() for s in self._sieves])
                * CachePlan(self._words).n_iterated_sums(
                    list(range(len(self._words)))
                  )
            )
        else:
            return (
                sum([s.nfeatures() for s in self._sieves])
                * len(self._words)
            )

    def _compile(self) -> None:
        # checks if the FruitBranch is configured correctly and ready
        # for fitting
        if not self._words:
            raise RuntimeError("No words specified for ISS calculation")
        if not self._sieves:
            raise RuntimeError("No FeatureSieve objects specified")

    def _collect_cache_keys(self) -> set[str]:
        # collects cache keys of all FitTransformers in the branch
        keys: set[str] = set()
        for prep in self._preparateurs:
            prep_keys = prep._get_cache_keys()
            if 'coquantile' in prep_keys:
                keys = keys.union(prep_keys['coquantile'])
        for sieve in self._sieves:
            sieve_keys = sieve._get_cache_keys()
            if 'coquantile' in sieve_keys:
                keys = keys.union(sieve_keys['coquantile'])
        return keys

    def _get_cache(self, X: np.ndarray) -> None:
        # returns the already processed cache needed in this branch
        self._cache = CoquantileCache()
        self._cache.process(X, list(self._collect_cache_keys()))

    def _select_fit_sample(self, X: np.ndarray) -> np.ndarray:
        # returns a sample of the data used for fitting
        if (isinstance(self._fit_sample_size, int)
                and self._fit_sample_size == 1):
            ind = np.random.randint(0, X.shape[0])
            return X[ind:ind+1, :, :]
        else:
            s = int(self._fit_sample_size * X.shape[0])
            if s < 1:
                s = 1
            indices = np.random.choice(X.shape[0], size=s, replace=False)
            return X[indices, :, :]

    def fit(self, X: np.ndarray) -> None:
        """Fits the branch to the given dataset. What this action
        explicitly does depends on the FruitBranch configuration.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            :meth:`~fruits.scope.force_input_shape`.
        :type X: np.ndarray
        """
        self._compile()

        self._get_cache(X)
        prepared_data = self._select_fit_sample(X)
        for prep in self._preparateurs:
            prep.fit(prepared_data)
            prepared_data = prep.transform(prepared_data, cache=self._cache)

        self._sieves_extended = []
        iss_calculations = SignatureCalculator().transform(
            prepared_data,
            words=self._words,
            **self._calculator_options
        )[0]
        for iterated_data in iss_calculations:
            iterated_data = iterated_data.reshape(iterated_data.shape[0]
                                                  * iterated_data.shape[1],
                                                  iterated_data.shape[2])
            sieves_copy = [sieve.copy() for sieve in self._sieves]
            for sieve in sieves_copy:
                sieve.fit(iterated_data[:, :])
            self._sieves_extended.append(sieves_copy)
        self._fitted = True

    def transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
    ) -> np.ndarray:
        """Transforms the given time series dataset. The results are
        the calculated features for the different time series.

        :param X: (Multidimensional) time series dataset as an array
            of three dimensions. Have a look at
            :meth:`~fruits.scope.force_input_shape`.
        :type X: np.ndarray
        :param callbacks: list of callbacks. To write your own callback,
            override the class
            :class:`~fruits.core.callback.AbstractCallback`.,
            defaults to []
        :type callbacks: list[AbstractCallback], optional
        :rtype: np.ndarray
        :raises: RuntimeError if ``self.fit`` wasn't called
        """
        if callbacks is None:
            callbacks = []
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")

        self._get_cache(X)
        prepared_data = force_input_shape(X)
        for prep in self._preparateurs:
            prepared_data = prep.transform(prepared_data, cache=self._cache)
            for callback in callbacks:
                callback.on_preparateur(prepared_data)
        for callback in callbacks:
            callback.on_preparation_end(prepared_data)

        sieved_data = np.zeros((prepared_data.shape[0],
                                self.nfeatures()))
        k = 0
        iss_calculations = SignatureCalculator().transform(
            prepared_data,
            words=self._words,
            **self._calculator_options
        )[0]
        for i, iterated_data in enumerate(iss_calculations):
            for callback in callbacks:
                callback.on_iterated_sum(iterated_data)
            for sieve in self._sieves_extended[i]:
                nf = sieve.nfeatures()
                new_features = nf * iterated_data.shape[1]
                for it in range(iterated_data.shape[1]):
                    sieved_data[:, k+it*nf:k+(it+1)*nf] = sieve.transform(
                        iterated_data[:, it, :],
                        cache=self._cache,
                    )
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
            `:meth:`~fruits.scope.force_input_shape`.
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
        copy_._calculator_options = self._calculator_options.copy()
        return copy_
