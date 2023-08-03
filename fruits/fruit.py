import inspect
from typing import Callable, Generator, Optional, Union

import numpy as np

from .cache import SharedSeedCache
from .callback import AbstractCallback
from .iss.iss import ISS
from .preparation.abstract import Preparateur
from .seed import Seed
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
        """Cuts the fruit and adds a new slice to the pipeline. The new
        slice will be switched to after creation.

        Args:
            slice (FruitSlice, optional): A slice to add to the fruit.
                If none is given, a new empty slice is added.
        """
        if slice is None:
            slice = FruitSlice()
        self._slices.append(slice)
        self._slc_index = len(self._slices) - 1
        self._fitted = False

    def copycut(self) -> None:
        """Cuts the fruit and adds a deepcopy of the currently selected
        slice to it. This method saves to write code for similar slices.
        The new slice will be switched to after creation.
        """
        self.cut(self.get_slice().deepcopy())

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

    def fit(
        self,
        X: np.ndarray,
        cache: Optional[SharedSeedCache] = None,
    ) -> None:
        """Fits all slices to the given data.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
        """
        cache_ = SharedSeedCache(X) if cache is None else cache
        for slc in self._slices:
            slc.fit(X, cache=cache_)
        self._fitted = True

    def transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
        cache: Optional[SharedSeedCache] = None,
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
        cache_ = SharedSeedCache(X) if cache is None else cache
        result = np.zeros((X.shape[0], self.nfeatures()))
        index = 0
        for slc in self._slices:
            for callback in callbacks:
                callback.on_next_slice()
            k = slc.nfeatures()
            result[:, index:index+k] = slc.transform(X, callbacks, cache_)
            index += k
        result = np.nan_to_num(result, copy=False, nan=0.0)
        return result

    def fit_transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
    ) -> np.ndarray:
        """Fits all slices to the given dataset and returns the
        transformed results of X from all slices.
        """
        self.fit(X)
        return self.transform(X, callbacks=callbacks)

    def summary(self) -> str:
        """Returns a summary of this object. The summary contains a
        summary for each slice in this fruit.
        """
        summary = 80*"=" + "\n"
        ident_string = "Fruit"
        if self.name != "":
            ident_string += f" {self.name!r}"
        ident_string += f" -> Features: {self.nfeatures()}"
        summary += "<" + f"{ident_string: ^78}" + ">\n"
        summary += 80*"=" + "\n"
        n = len(self._slices)
        if n%2 != 0:
            n -= 1
        for islc in range(0, n, 2):
            summary += "|" + 38*"-" + "||" + 38*"-" + "|\n"
            left = self._slices[islc].summary().split("\n")
            right = self._slices[islc+1].summary().split("\n")
            left += [38*" " for _ in range(len(right)-len(left))]
            right += [38*" " for _ in range(len(left)-len(right))]
            summary += "\n".join(f"|{l}||{r}|" for l, r in zip(left, right))
            summary += "\n|" + 38*"-" + "||" + 38*"-" + "|\n"
        if len(self._slices)%2 != 0:
            summary += "|" + 38*"-" + "|\n|"
            summary += self._slices[-1].summary().replace("\n", "|\n|")
            summary += "|\n|" + 38*"-" + "|\n"
        summary += f"{'':=^80}"
        return summary

    def copy(self) -> "Fruit":
        """Creates a shallow copy of this Fruit by copying all slices."""
        copy_ = Fruit(self.name + " (Copy)")
        for slc in self._slices:
            copy_.cut(slc.copy())
        return copy_

    def deepcopy(self) -> "Fruit":
        """Creates a deep copy of this Fruit by deep copying all slices."""
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
        self._iss: list[ISS] = []
        self._sieves: list[FeatureSieve] = []

        # list with inner lists containing sieves
        # all sieves in one list are trained on an iterated sums of one
        # word (the second inner list iterates over extended letters if
        # the iss mode is set to extended)
        self._sieves_extended: list[list[FeatureSieve]] = []

        # configurations for fitting
        self._fitted: bool = False
        self.fit_sample_size: Union[float, int] = 1

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

    def add_iss(self, iss: ISS) -> None:
        """Adds a iss to this fruit slice."""
        if not isinstance(iss, ISS):
            raise TypeError
        self._iss.append(iss)
        self._fitted = False

    def get_iss(self) -> list[ISS]:
        """Returns a list of all iss in this fruit slice."""
        return self._iss

    def clear_iss(self) -> None:
        """Removes all iss in this fruit slice."""
        self._iss = []
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

                - :class:`~fruits.preparation.abstract.Preparateur`
                - :class:`~fruits.iss.iss.ISS`
                - :class:`~fruits.sieving.abstract.FeatureSieve`
        """
        for obj in objects:
            if inspect.isclass(obj):
                obj = obj()

            if isinstance(obj, Preparateur):
                self.add_preparateur(obj)
            elif isinstance(obj, ISS):
                self.add_iss(obj)
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
        self.clear_iss()
        self.clear_sieves()
        self.iss_mode = "single"
        self.fit_sample_size = 1

    def nfeatures(self) -> int:
        """Returns the total number of features the current
        configuration produces.
        """
        return int(
            sum(s.nfeatures() for s in self._sieves)
            * np.prod([iss.n_iterated_sums() for iss in self._iss])
        )

    def _compile(self) -> None:
        # checks if the FruitSlice is configured correctly and ready
        # for fitting
        if not self._iss:
            raise RuntimeError("No ISS given")
        if not self._sieves:
            raise RuntimeError("No feature sieves given")

    def _select_fit_sample(self, X: np.ndarray) -> np.ndarray:
        # returns a sample of the data used for fitting
        if (isinstance(self.fit_sample_size, int)
                and self.fit_sample_size == 1):
            ind = np.random.randint(0, X.shape[0])
            return X[ind:ind+1, :, :]
        s = max(int(self.fit_sample_size * X.shape[0]), 1)
        indices = np.random.choice(X.shape[0], size=s, replace=False)
        return X[indices, :, :]

    def _iterate_iss(
        self,
        X: np.ndarray,
        iss_index: int = 0,
    ) -> Generator[np.ndarray, None, None]:
        # iteratively calculate all iterated sums from all ISS
        if iss_index == len(self._iss):
            yield X[:, 0, :]
        else:
            for itsums in self._iss[iss_index].batch_transform(X):
                for itsum in itsums:
                    yield from self._iterate_iss(
                        itsum[:, np.newaxis, :],
                        iss_index+1,
                    )

    def fit(
        self,
        X: np.ndarray,
        cache: Optional[SharedSeedCache] = None,
    ) -> None:
        """Fits the slice to the given dataset. What this action
        explicitly does depends on its configuration.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
        """
        self._compile()

        if cache is None:
            cache = SharedSeedCache(X)
        prepared_data = self._select_fit_sample(X)
        for prep in self._preparateurs:
            prep._cache = cache
            prep.fit(prepared_data)
            prepared_data = prep.transform(prepared_data)

        for iss in self._iss:
            iss._cache = cache
            if iss.requires_fitting:
                iss.fit(X)

        if not any(sieve.requires_fitting for sieve in self._sieves):
            self._fitted = True
            return

        self._sieves_extended = []

        for itsum in self._iterate_iss(prepared_data):
            sieves_copy = [sieve.copy() for sieve in self._sieves]
            for sieve in sieves_copy:
                sieve._cache = cache
                sieve.fit(itsum)
            self._sieves_extended.append(sieves_copy)
        self._fitted = True

    def transform(
        self,
        X: np.ndarray,
        callbacks: Optional[list[AbstractCallback]] = None,
        cache: Optional[SharedSeedCache] = None,
    ) -> np.ndarray:
        """Transforms the given time series dataset. The results are
        the calculated features for the different time series.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
            callbacks: A list of callbacks. To write your own callback,
                inherit from :class:`~fruits.callback.AbstractCallback`.
                Defaults to None.

        Raises:
            RuntimeError: If :meth:`fit` wasn't called.
        """
        if callbacks is None:
            callbacks = []
        if not self._fitted:
            raise RuntimeError("Missing call of self.fit")

        if cache is None:
            cache = SharedSeedCache(X)
        prepared_data = X
        for prep in self._preparateurs:
            prep._cache = cache
            prepared_data = prep.transform(prepared_data)
            for callback in callbacks:
                callback.on_preparateur(prepared_data)
        for callback in callbacks:
            callback.on_preparation_end(prepared_data)

        sieved_data = np.zeros((prepared_data.shape[0], self.nfeatures()))
        for iss in self._iss:
            iss._cache = cache
        k = 0
        for i, itsum in enumerate(self._iterate_iss(prepared_data)):
            for callback in callbacks:
                callback.on_iterated_sum(itsum)
            sieves = self._sieves_extended[i] if self._sieves_extended else (
                self._sieves
            )
            for sieve in sieves:
                sieve._cache = cache
                nf = sieve.nfeatures()
                sieved_data[:, k:k+nf] = sieve.transform(itsum)
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
        summary = f"{f'FruitSlice -> {self.nfeatures()}': ^38}"
        summary += "\n" + 38*"-" + "\n"
        summary += f"{f'Preparateurs ({len(self._preparateurs)}):': <38}"
        summary += "\n"
        summary += f"\n".join(
            f"{f'    + {x}': <38}" for x in self._preparateurs
        )
        if len(self._preparateurs) == 0:
            summary += 38*" "
        summary += "\n"
        summary += f"{f'ISS Calculators ({len(self._iss)}):': <38}"
        if len(self._iss) == 0:
            summary += 38*" "
        for iss in self._iss:
            summary += f"\n{f'    + {iss} -> {iss.n_iterated_sums()}': <38}"
            summary += f"\n{f'       | words: {len(iss.words)}': <38}"
            semiring = iss.semiring.__class__.__name__
            summary += f"\n{f'       | semiring: {semiring}': <38}"
            weighting = "None" if iss.weighting is None else (
                iss.weighting.__class__.__name__
            )
            summary += f"\n{f'       | weighting: {weighting}': <38}"
        if len(self._iss) == 0:
            summary += "\n"
        summary += f"\n{f'Sieves ({len(self._sieves)}):': <38}"
        if len(self._sieves) == 0:
            summary += "\n" + 38*" "
        for sv in self._sieves:
            name = sv.__class__.__name__
            summary += f"\n{f'    + {name} -> {sv.nfeatures()}': <38}"
        return summary

    def copy(self) -> "FruitSlice":
        """Returns a shallow copy of this FruitSlice with same settings
        but all calculation progress erased.
        """
        copy_ = FruitSlice()
        for preparateur in self._preparateurs:
            copy_.add(preparateur)
        for iss in self._iss:
            copy_.add(iss)
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
        for iss in self._iss:
            copy_.add(iss.copy())
        for sieve in self._sieves:
            copy_.add(sieve.copy())
        copy_.fit_sample_size = self.fit_sample_size
        return copy_
