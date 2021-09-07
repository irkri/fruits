from typing import List, Union

import numpy as np

from fruits.base.scope import force_input_shape, check_input_shape
from fruits.core.wording import Word, SimpleWord
from fruits.core.backend import _fast_ISS, _slow_ISS

class CachePlan:
    """Class that creates a plan for the efficient calculation of
    iterated sums using the given words. This plan is needed when the
    mode of an ISSCalculator is set to "extended".
    The plan removes repetition in calculation.
    """
    def __init__(self, words: List[Word]):
        self._words = words

        self._create_plan()

    def _create_plan(self):
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

        :type index: int
        :rtype: int
        """
        return self._plan[index]

    def n_iterated_sums(self, word_indices: List[int]) -> int:
        """Returns the number of iterated sums that have to be
        calculated using this plan for the words with the given indices.

        :rtype: int
        """
        return sum(self._plan[i] for i in word_indices)


class ISSCalculator:
    """Class that is responsible for managing the calculation of
    iterated sums.

    :param mode: Mode used for the calculation. Has to be either 'single'
        or 'extended'. It is a public property of the calculator.
        Specifying the mode used in a :class:`~fruits.base.fruit.Fruit`
        can be done by setting ``fruit.calculator.mode``.,
        defaults to "single"
    :type mode: str, optional
    """
    def __init__(self, mode: str = "single"):
        self.mode = mode
        self._batch_size = -1
        self._words = None
        self._X = None
        self._cache_plan = None
        self._started = False

    @property
    def mode(self) -> str:
        """Mode of the object that has to be one of the following
        values.

        - 'single': Calculates one iterated sum for each given word.
        - 'extended': For each given word, the iterated sum for each
            sequential combination of extended letters in that word will
            be calculated. So for a simple word like ``[21][121][1]``
            the calculator returns the iterated sums for ``[21]``,
            ``[21][121]`` and ``[21][121][1]``.

        :rtype: str
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if not mode in {"single", "extended"}:
            raise ValueError("Unknown mode supplied")
        self._mode = mode

    @property
    def batch_size(self) -> int:
        """Number of words for which the iterated sums are calculated at
        once when starting the iterator of this calculator.
        This doesn't have to be the same number as the actual number of
        iterated sums returned. Default value is -1, which means the
        results of all words are given at once.
        """
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def _transform_simple_word(self, word: SimpleWord) -> np.ndarray:
        # transforms all simplewords for faster calculation with a
        # backend function
        simple_word_raw = [el for el in word]
        word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
        for i in range(len(simple_word_raw)):
            for j in range(len(simple_word_raw[i])):
                word_transformed[i, j] = simple_word_raw[i][j]
        return word_transformed

    def _n_iterated_sums(self, words: List[Word]) -> int:
        if self.mode == "extended":
            return CachePlan(words).n_iterated_sums(list(range(len(words))))
        else:
            return len(words)

    def start(self, X: np.ndarray, words: List[Word]):
        self._started = True
        self._X = X
        if not check_input_shape(X):
            self._X = force_input_shape(X)

        self._words = words
        if self.mode == "extended":
            self._cache_plan = CachePlan(words)

    def __iter__(self):
        if not self._started:
            raise RuntimeError("Calculator not started yet")
        self._current_word = 0
        self._real_batch_size = len(self._words) if self._batch_size == -1 \
                                                 else self._batch_size
        return self

    def __next__(self) -> np.ndarray:
        if self._current_word >= len(self._words):
            raise StopIteration
        if self.mode == "extended":
            results = np.zeros((self._X.shape[0],
                                self._cache_plan.n_iterated_sums(
                                    list(range(self._current_word, 
                                    self._current_word + \
                                    self._real_batch_size))),
                                self._X.shape[2]))
        else:
            results = np.zeros((self._X.shape[0],
                                self._real_batch_size,
                                self._X.shape[2]))

        index = 0
        for i in range(self._current_word,
                       self._current_word+self._real_batch_size):
            if i > len(self._words):
                self._current_word = len(self._words)
                return results[:, :i, :]
            ext = 1
            if self.mode == "extended":
                ext = self._cache_plan.unique_el_depth(i)

            if isinstance(self._words[i], SimpleWord):
                results[:, index:index+ext, :] = _fast_ISS(self._X,
                    self._transform_simple_word(self._words[i]),
                    np.array([0] + self._words[i].alpha + [0],
                             dtype=np.float32),
                    ext)
            elif isinstance(self._words[i], Word):
                results[:, index:index+ext, :] = _slow_ISS(self._X,
                    self._words[i],
                    np.array([0] + self._words[i].alpha + [0],
                             dtype=np.float32),
                    ext)
            else:
                raise TypeError(f"Unknown word type: {type(self._words[i])}")
            index += ext

        self._current_word += self._real_batch_size
        return results

    def copy(self) -> "ISSCalculator":
        """Returns a copy of this calculator.

        :rtype: ISSCalculator
        """
        calc = ISSCalculator(mode=self.mode)
        return calc


def ISS(X: np.ndarray,
        words: Union[List[Word], Word],
        mode: str = "single") -> np.ndarray:
    """Takes in a number of time series and a list of words and
    calculates the iterated sums for each time series in ``X``. This
    function is just used as wrapper for the class
    :class:`~fruits.core.iss.ISSCalculator`. For more information on the
    calculation of the iterated sums signature, have a look at the
    calculator.

    :param X: Three dimensional numpy array containing a
        multidimensional time series dataset.
    :type X: numpy.ndarray
    :type words: Union[List[Word], Word]
    :param mode: Mode of the used calculator. Has to be either "single"
        or "extended".
    :type mode: str
    :returns: Numpy array of shape
        ``(X.shape[0], len(words), X.shape[2])``.
    :rtype: numpy.ndarray
    """
    words = [words] if isinstance(words, Word) else words

    calculator = ISSCalculator(mode)
    calculator.start(X, words)

    return list(calculator)[0]
