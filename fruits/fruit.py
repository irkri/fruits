import inspect
from typing import Any, Callable, Literal, Optional, Union

import numpy as np

from .callback import AbstractCallback
from .iss.iss import ISS, CachePlan
from .iss.words.word import Word
from .preparation.abstract import Preparateur
from .seed import Seed
from .cache import SharedSeedCache
from .sieving.abstract import FeatureSieve


class Fruit:
    """Feature Extractor using iterated sums.

    A Fruit consists of at least one slice <:class:`FruitSlice`>.
    A slice can be customized by adding preparateurs, words or sieves.

    A simple example (using two slices):

    .. code-block:: python

        fruit = fruits.Fruit("My Fruit")
        # optional: add preparateurs for preprocessing
        fruit.add(fruits.preparation.INC)
        # add words for iterated sums calculation
        fruit.add(*fruits.words.of_weight(4))
        # choose sieves
        fruit.add(fruits.sieving.PPV(0.5))
        fruit.add(fruits.sieving.END)

        # add a new slice without INC
        fruit.cut()
        fruit.add(*fruits.words.of_weight(4))
        fruit.add(fruits.sieving.PPV(0.5))
        fruit.add(fruits.sieving.END)

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

    def __init__(self, name: str = "") -> None:
        self.name: str = name
        # list of FruitSlicees
        self._slices: list[FruitSlice] = []
        # pointer for the current slice index
        self._slc_index: int = 0
        self._fitted: bool = False
        self._iterator_index: int = -1

    def cut(self, slice: Optional["FruitSlice"] = None) -> None:
        """Cuts the fruit and adds a new slice to the pipeline.

        Args:
            slice (FruitSlice, optional): A slice to add to the fruit.
                If none is given, a new empty slice is added.
        """
        if slice is None:
            slice = FruitSlice()
        self._slices.append(slice)
        self._slc_index = len(self._slices) - 1
        self._fitted = False

    def get_slice(self, index: Optional[int] = None) -> "FruitSlice":
        """Returns the currently selected slice or the slice
        corresponding to the supplied index.
        """
        if index is None:
            return self._slices[self._slc_index]
        return self._slices[index]

    def switch_slice(self, index: int) -> None:
        """Switches to the slice with the given index.

        Args:
            index (int): Integer in ``[0, 1, ..., len(self)-1]``.
        """
        if not (0 <= index < len(self._slices)):
            raise IndexError("Index has to be in [0, len(self)-1]")
        self._slc_index = index

    def add(self, *objects: Union[Seed, Callable[[], Seed]]) -> None:
        """Adds one or multiple object(s) to the currently selected
        slice.

        Args:
            objects: One or more objects of the
                following types:

                - :class:`~fruits.preparation.abstract.DataPreparateur`
                - :class:`~fruits.words.word.Word`
                - :class:`~fruits.sieving.abstract.FeatureSieve`
        """
        if len(self._slices) == 0:
            self.cut()
        self._slices[self._slc_index].add(*objects)
        self._fitted = False

    def nfeatures(self) -> int:
        """Returns the total sum of features in all slices."""
        return sum(slc.nfeatures() for slc in self._slices)

    def configure(self, **kwargs: Any) -> None:
        """Calls ``slc.configure(**kwargs)`` for each FruitSlice
        ``slc`` in the Fruit with the specified arguments.

        Args:
            iss_mode (str, optional): Mode of the ISS calculator.
                Following options are available.

                - 'single':
                    Calculates one iterated sum for each given word.
                    Default behaviour.
                - 'extended':
                    For each given word, the iterated sum for
                    each sequential combination of extended letters
                    in that word will be calculated. So for a simple
                    word like ``[21][121][1]`` the calculator
                    returns the iterated sums for ``[21]``,
                    ``[21][121]`` and ``[21][121][1]``.
            fit_sample_size (float or int, optional): Size of the random
                time series sample that is used for fitting. This is
                represented as a float which will be multiplied by
                ``X.shape[0]`` or ``1`` for one random time series.
                Defaults to 1.
        """
        for slc in self._slices:
            slc.configure(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        """Fits all slices to the given data.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
        """
        for slc in self._slices:
            slc.fit(X)
        self._fitted = True

    def transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
    ) -> np.ndarray:
        """Returns a two dimensional array of all features from all
        slices this Fruit object contains.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
            callbacks: A list of callbacks. To write your own callback,
                override the class
                :class:`~fruits.callback.AbstractCallback`.
                Defaults to None.

        Raises:
            RuntimeError: If :meth:`fit` wasn't called.
        """
        if callbacks is None:
            callbacks = []
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for slc in self._slices:
            for callback in callbacks:
                callback.on_next_slice()
            k = slc.nfeatures()
            result[:, index:index+k] = slc.transform(X, callbacks)
            index += k
        result = np.nan_to_num(result, copy=False, nan=0.0)
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits all slices to the given dataset and returns the
        transformed results of X from all slices.
        """
        self.fit(X)
        return self.transform(X)

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains a
        summary for each FruitSlice in this Fruit object.
        """
        summary = f"{f'Summary of fruits.Fruit: {self.name}':=^80}"
        summary += f"\nSlices: {len(self)}"
        summary += f"\nFeatures: {self.nfeatures()}"
        for slc in self:
            summary += "\n\n" + slc.summary()
        summary += f"\n{'End of Summary':=^80}"
        return summary

    def copy(self) -> "Fruit":
        """Creates a shallow copy of this Fruit by copying all slices.
        """
        copy_ = Fruit(self.name + " (Copy)")
        for slc in self._slices:
            copy_.cut(slc.copy())
        return copy_

    def deepcopy(self) -> "Fruit":
        """Creates a deep copy of this Fruit by deep copying all slices.
        """
        copy_ = Fruit(self.name + " (Deepcopy)")
        for slc in self._slices:
            copy_.cut(slc.deepcopy())
        return copy_

    def __len__(self) -> int:
        return len(self._slices)

    def __iter__(self) -> "Fruit":
        self._iterator_index = -1
        return self

    def __next__(self) -> "FruitSlice":
        if self._iterator_index < len(self._slices)-1:
            self._iterator_index += 1
            return self._slices[self._iterator_index]
        raise StopIteration()

    def __getitem__(self, index: int) -> "FruitSlice":
        return self.get_slice(index)


class FruitSlice:
    """One slice of a Fruit.

    A FruitSlice object extracts values from time series data that are
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
            the slice will be fitted on the iterated sums from the
            previous step. The slice then returns an array of numbers
            (the transformed results from those sieves), i.e. the
            features for each time series.
    """

    def __init__(self) -> None:
        # lists of used classes for data processing
        self._preparateurs: list[Preparateur] = []
        self._words: list[Word] = []
        self._sieves: list[FeatureSieve] = []

        # options for the ISS calculation
        self.iss_mode: Literal['single', 'extended'] = "single"

        # list with inner lists containing sieves
        # all sieves in one list are trained on one specific output
        # of an ISS-result
        self._sieves_extended: list[list[FeatureSieve]] = []

        # configurations for fitting
        self._fitted: bool = False
        self.fit_sample_size: Union[float, int] = 1

    def configure(
        self,
        *,
        iss_mode: Literal['single', 'extended'] = "single",
        fit_sample_size: Union[float, int] = 1,
    ) -> None:
        """Makes changes to the default configuration of a fruit slice
        if arguments differ from ``None``.

        Args:
            iss_mode (str, optional): Mode of the ISS calculator.
                Following options are available.

                - 'single':
                    Calculates one iterated sum for each given word.
                    Default behaviour.
                - 'extended':
                    For each given word, the iterated sum for
                    each sequential combination of extended letters
                    in that word will be calculated. So for a simple
                    word like ``[21][121][1]`` the calculator
                    returns the iterated sums for ``[21]``,
                    ``[21][121]`` and ``[21][121][1]``.
            fit_sample_size (float or int, optional): Size of the random
                time series sample that is used for fitting. This is
                represented as a float which will be multiplied by
                ``X.shape[0]`` or ``1`` for one random time series.
                Defaults to 1.
        """
        self.iss_mode = iss_mode
        self.fit_sample_size = fit_sample_size

    def add_preparateur(self, preparateur: Preparateur) -> None:
        """Adds a preparateur to the fruit slice."""
        if not isinstance(preparateur, Preparateur):
            raise TypeError
        self._preparateurs.append(preparateur)
        self._fitted = False

    def get_preparateurs(self) -> list[Preparateur]:
        """Returns a list of all preparateurs in this fruit slice."""
        return self._preparateurs

    def clear_preparateurs(self) -> None:
        """Removes all preparateurs in this fruit slice."""
        self._preparateurs = []
        self._fitted = False

    def add_word(self, word: Word) -> None:
        """Adds a word to this fruit slice."""
        if not isinstance(word, Word):
            raise TypeError
        self._words.append(word)
        self._fitted = False

    def get_words(self) -> list[Word]:
        """Returns a list of all words in this fruit slice."""
        return self._words

    def clear_words(self) -> None:
        """Removes all words in this fruit slice."""
        self._words = []
        self._sieves_extended = []
        self._fitted = False

    def add_sieve(self, sieve: FeatureSieve) -> None:
        """Appends a new feature sieve to this fruit slice."""
        if not isinstance(sieve, FeatureSieve):
            raise TypeError
        self._sieves.append(sieve)
        self._fitted = False

    def get_sieves(self) -> list[FeatureSieve]:
        """Returns a list of all feature sieves in this fruit slice."""
        return self._sieves

    def clear_sieves(self) -> None:
        """Removes all feature sieves in this fruit slice."""
        self._sieves = []
        self._sieves_extended = []
        self._fitted = False

    def add(self, *objects: Union[Seed, Callable[[], Seed]]) -> None:
        """Adds one or multiple object(s) to this fruit slice.

        Args:
            objects: One or more objects of the following types:

                - :class:`~fruits.preparation.abstract.DataPreparateur`
                - :class:`~fruits.words.word.Word`
                - :class:`~fruits.sieving.abstract.FeatureSieve`
        """
        for obj in objects:
            if inspect.isclass(obj):
                obj = obj()

            if isinstance(obj, Preparateur):
                self.add_preparateur(obj)
            elif isinstance(obj, Word):
                self.add_word(obj)
            elif isinstance(obj, FeatureSieve):
                self.add_sieve(obj)
            else:
                raise TypeError(f"Cannot add variable of type {type(obj)}")

    def clear(self) -> None:
        """Clears all settings, configurations and calculated results
        in this fruit slice.

        After the slice is cleared, it has the same settings as a newly
        created :class:`FruitSlice` object.
        """
        self.clear_preparateurs()
        self.clear_words()
        self.clear_sieves()
        self.iss_mode = "single"
        self.fit_sample_size = 1

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.
        """
        if self.iss_mode == "extended":
            return (
                sum(s.nfeatures() for s in self._sieves)
                * CachePlan(self._words).n_iterated_sums(
                    list(range(len(self._words)))
                  )
            )
        return sum(s.nfeatures() for s in self._sieves) * len(self._words)

    def _compile(self) -> None:
        # checks if the FruitSlice is configured correctly and ready
        # for fitting
        if not self._words:
            raise RuntimeError("No words specified for ISS calculation")
        if not self._sieves:
            raise RuntimeError("No FeatureSieve objects specified")

    def _select_fit_sample(self, X: np.ndarray) -> np.ndarray:
        # returns a sample of the data used for fitting
        if (isinstance(self.fit_sample_size, int)
                and self.fit_sample_size == 1):
            ind = np.random.randint(0, X.shape[0])
            return X[ind:ind+1, :, :]
        s = max(int(self.fit_sample_size * X.shape[0]), 1)
        indices = np.random.choice(X.shape[0], size=s, replace=False)
        return X[indices, :, :]

    def fit(self, X: np.ndarray) -> None:
        """Fits the slice to the given dataset. What this action
        explicitly does depends on its configuration.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
        """
        self._compile()

        cache = SharedSeedCache()
        prepared_data = self._select_fit_sample(X)
        for prep in self._preparateurs:
            prep._cache = cache
            prep.fit(prepared_data)
            prepared_data = prep.transform(prepared_data)

        self._sieves_extended = []
        iss_calculations = ISS(
            prepared_data,
            words=self._words,
            mode=self.iss_mode,
            batch_size=1,
        )
        for iterated_data in iss_calculations:
            iterated_data = iterated_data[:, 0, :]
            sieves_copy = [sieve.copy() for sieve in self._sieves]
            for sieve in sieves_copy:
                sieve._cache = cache
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

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
            callbacks: A list of callbacks. To write your own callback,
                override the class
                :class:`~fruits.callback.AbstractCallback`.
                Defaults to None.

        Raises:
            RuntimeError: If :meth:`fit` wasn't called.
        """
        if callbacks is None:
            callbacks = []
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")

        prepared_data = X
        for prep in self._preparateurs:
            prepared_data = prep.transform(prepared_data)
            for callback in callbacks:
                callback.on_preparateur(prepared_data)
        for callback in callbacks:
            callback.on_preparation_end(prepared_data)

        sieved_data = np.zeros((prepared_data.shape[0],
                                self.nfeatures()))
        k = 0
        iss_calculations = ISS(
            prepared_data,
            words=self._words,
            mode=self.iss_mode,
            batch_size=1,
        )
        for i, iterated_data in enumerate(iss_calculations):
            for callback in callbacks:
                callback.on_iterated_sum(iterated_data)
            for sieve in self._sieves_extended[i]:
                nf = sieve.nfeatures()
                sieved_data[:, k:k+nf] = sieve.transform(
                    iterated_data[:, 0, :]
                )
                for callback in callbacks:
                    callback.on_sieve(sieved_data[k:k+nf])
                k += nf
        for callback in callbacks:
            callback.on_sieving_end(sieved_data)
        return sieved_data

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """This function does the same that calling ``self.fit(X)`` and
        ``self.transform(X)`` consecutively does.
        """
        self.fit(X)
        return self.transform(X)

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains all
        added preparateurs, words and sieves.
        """
        summary = f"{'fruits.FruitSlice':-^80}"
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

    def copy(self) -> "FruitSlice":
        """Returns a shallow copy of this FruitSlice with same settings
        but all calculation progress erased.
        """
        copy_ = FruitSlice()
        for preparateur in self._preparateurs:
            copy_.add(preparateur)
        for iterator in self._words:
            copy_.add(iterator)
        for sieve in self._sieves:
            copy_.add(sieve)
        return copy_

    def deepcopy(self) -> "FruitSlice":
        """Returns a deep copy of this FruitSlice with same settings
        but all calculation progress erased.
        """
        copy_ = FruitSlice()
        for preparateur in self._preparateurs:
            copy_.add(preparateur.copy())
        for iterator in self._words:
            copy_.add(iterator.copy())
        for sieve in self._sieves:
            copy_.add(sieve.copy())
        copy_.iss_mode = self.iss_mode
        copy_.fit_sample_size = self.fit_sample_size
        return copy_
