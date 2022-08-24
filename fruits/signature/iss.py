from typing import Union

import numpy as np

from fruits.words.word import Word, SimpleWord
from fruits.signature.backend import _fast_ISS, _slow_ISS


class CachePlan:
    """Class that creates a plan for the efficient calculation of
    iterated sums using the given words. This plan is needed when the
    mode of an :class:`~fruits.signature.iss.SignatureCalculator` is set
    to "extended". The plan removes repetition in calculation.
    """

    def __init__(self, words: list[Word]) -> None:
        self._words = words
        self._create_plan()

    def _create_plan(self) -> None:
        # prefix search in the word strings generates a cache plan
        self._plan = []
        word_strings = [str(word) for word in self._words]
        for i, wstr in enumerate(word_strings):
            els = wstr.split("[")[1:]
            # start variable used for speed up
            start = 0
            depth = len(els)
            for j in range(len(els)):
                for k in range(start, i):
                    if word_strings[k].startswith("["+"[".join(els[:j+1])):
                        # a prefix in word k is found -> ignore the prefix
                        start = k
                        depth -= 1
                        break
                else:
                    # one prefix didn't exist -> next larger prefix will
                    # not exist either
                    break
            self._plan.append(depth)

    def unique_el_depth(self, index: int) -> int:
        """Returns the total number of iterated sums to be calculated
        for the word with the given index.
        """
        return self._plan[index]

    def n_iterated_sums(self, word_indices: list[int]) -> int:
        """Returns the number of iterated sums that have to be
        calculated using this plan for the words with the given indices.
        """
        return sum(self._plan[i] for i in word_indices)


class SignatureCalculation:
    """Object type that is returned by the ``transform()`` method of a
    :class:`~fruits.signature.iss.SignatureCalculator`.
    This object is an iterable. The elements in the iterator are numpy
    arrays containing the results of iterated sums. The number of
    results in one of these arrays is determined by the option
    ``batch_size``.
    """

    def __init__(
        self,
        X: np.ndarray,
        words: list[Word],
        mode: str = "single",
        batch_size: int = -1,
    ) -> None:
        self._X = X
        self._words = words
        if mode not in ["single", "extended"]:
            raise ValueError("mode can either be 'single' or 'extended'")
        self._mode = mode
        self._batch_size = batch_size
        self._cache_plan: CachePlan
        self._real_batch_size: int = 0

    def _transform_simple_word(self, word: SimpleWord) -> np.ndarray:
        # transforms all simplewords for faster calculation with a
        # backend function
        simple_word_raw = [el for el in word]
        word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
        for i in range(len(simple_word_raw)):
            for j in range(len(simple_word_raw[i])):
                word_transformed[i, j] = simple_word_raw[i][j]
        return word_transformed

    def _n_iterated_sums(self, words: list[Word]) -> int:
        # returns the number of iterated sums this object produces
        if self._mode == "extended":
            return CachePlan(words).n_iterated_sums(list(range(len(words))))
        else:
            return len(words)

    def _get_alpha(self, word: Word) -> np.ndarray:
        # returns a better format of the weighting of the given word
        return np.array([0.0] + word.alpha + [0.0], dtype=np.float32)

    def __iter__(self) -> "SignatureCalculation":
        if self._mode == "extended":
            self._cache_plan = CachePlan(self._words)
        self._current_word = 0
        self._real_batch_size = self._batch_size
        if self._batch_size == -1:
            self._real_batch_size = len(self._words)
        return self

    def __next__(self) -> np.ndarray:
        if self._current_word >= len(self._words):
            raise StopIteration
        nsums = self._real_batch_size
        if self._mode == "extended":
            nsums = self._cache_plan.n_iterated_sums(
                list(range(self._current_word,
                           self._current_word + self._real_batch_size))
            )
        results = np.zeros((self._X.shape[0], nsums, self._X.shape[2]))

        index = 0
        for i in range(self._current_word,
                       self._current_word+self._real_batch_size):
            if i > len(self._words):
                self._current_word = len(self._words)
                return results[:, :i, :]
            ext = 1
            if self._mode == "extended":
                ext = self._cache_plan.unique_el_depth(i)

            alphas = self._get_alpha(self._words[i])

            if isinstance(self._words[i], SimpleWord):
                results[:, index:index+ext, :] = _fast_ISS(
                    self._X,
                    self._transform_simple_word(self._words[i]),
                    alphas,
                    ext,
                )
            elif isinstance(self._words[i], Word):
                results[:, index:index+ext, :] = _slow_ISS(
                    self._X,
                    self._words[i],
                    alphas,
                    ext,
                )
            else:
                raise TypeError(f"Unknown word type: {type(self._words[i])}")
            index += ext

        self._current_word += self._real_batch_size
        return results


class SignatureCalculator:
    """Class that is responsible for managing the calculation of
    iterated sums.
    """

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the calculator on the given time series dataset."""

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Starts and returns an iterated sums signature calculation.

        Args:
            X (np.ndarray): Input time series dataset.
            kwargs: Has to include the argument ``words`` as a list
                of :class:`~fruits.signature.wording.Word` objects.
                Possible optional arguments are:

                - ``batch_size``: Batch size of the calculaor.
                    Number of words for which the iterated sums are
                    calculated at once when starting the iterator of
                    this calculator. This doesn't have to be the same
                    number as the actual number of iterated sums
                    returned. Default value is -1, which means the
                    results of all words are given at once.
                - ``mode``: Mode of the calculator.
                    Following options are available.

                    - 'single':
                        Calculates one iterated sum for each given word.
                    - 'extended':
                        For each given word, the iterated sum for
                        each sequential combination of extended letters
                        in that word will be calculated. So for a simple
                        word like ``[21][121][1]`` the calculator
                        returns the iterated sums for ``[21]``,
                        ``[21][121]`` and ``[21][121][1]``.

        Returns:
            A numpy array with only one object of type
            :class:`~fruits.signature.iss.SignatureCalculation`.
        """
        return np.array(
            [SignatureCalculation(X, **kwargs)],
            dtype=object
        )


def ISS(
    X: np.ndarray,
    words: Union[list[Word], Word],
    mode: str = "single",
) -> np.ndarray:
    """Takes in a number of time series and a list of words and
    calculates the iterated sums for each time series in ``X``. This
    function is just used as a convenience wrapper of the class
    :class:`~fruits.signature.iss.SignatureCalculator`.
    For more information on the calculation of the iterated sums
    signature, have a look at the calculator.

    Args:
        X (np.ndarray): Three dimensional numpy array containing a
            multidimensional time series dataset.
        words (one or a list of Words): Words to calculate the ISS for.
        mode (str): Mode of the used calculator. Has to be either
            "single" or "extended".

    Returns
        Numpy array of shape ``(X.shape[0], len(words), X.shape[2])``.
    """
    words = [words] if isinstance(words, Word) else words

    calculator = SignatureCalculator()

    result_iterator = calculator.transform(
        X,
        words=words,
        mode=mode,
        batch_size=-1,
    )[0]
    return next(iter(result_iterator))
