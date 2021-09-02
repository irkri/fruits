from typing import List, Union

import numpy as np

from fruits.base.scope import force_input_shape, check_input_shape
from fruits.core.wording import Word, SimpleWord
from fruits.core.backend import _fast_ISS, _slow_ISS

class ISSCalculator:
    """Class that is responsible for managing the calculation of
    iterated sums.

    :param mode: Mode used for the calculation. Has to be either 'whole'
        or 'extended'. It is a public property of the calculator.
        Specifying the mode used in a :class:`~fruits.base.fruit.Fruit`
        can be done by setting ``fruit.calculator.mode``.,
        defaults to "whole"
    :type mode: str, optional
    """
    def __init__(self, mode: str = "whole"):
        self.mode = mode

    @property
    def mode(self) -> str:
        """Mode of the object that has to be one of the following
        values.

        - 'whole': Calculates one iterated sum for each given word.
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
        if not mode in {"whole", "extended"}:
            raise ValueError("Unknown mode supplied")
        self._mode = mode

    def _split_words(self, words: List[Word]):
        # splits the list of words into seperate ones for simple and 
        # complex words and saves the indices of their original
        # position in the given list
        simple_words = []
        general_words = []
        simple_words_index = []
        general_words_index = []
        if len(words) == 0:
            raise ValueError("At least one word for ISS calculation needed")
        for i, word in enumerate(words):
            if isinstance(word, SimpleWord):
                simple_words.append(word)
                simple_words_index.append(i)
            elif isinstance(word, Word):
                general_words.append(word)
                general_words_index.append(i)
            else:
                raise TypeError("Given words have to be objects of a "
                                "class that inherits from Word")
        return (simple_words,
                simple_words_index,
                general_words,
                general_words_index)

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
        # returns the number of iterated sums to be calulated
        s = 0
        for word in words:
            if self.mode == "extended":
                s += len(word)
            else:
                s += 1
        return s

    def calculate(self, X: np.ndarray, words: List[Word]) -> np.ndarray:
        """Does the already initilialized calculation and returns the
        results.

        :param X: Input time series dataset.
        :type X: np.ndarray
        :param words: Words used for the calculation.
        :type: words: List[Words]
        :rtype: np.ndarray
        """
        if not check_input_shape(X):
            X = force_input_shape(X)
        simple_words, simple_words_ind, general_words, general_words_ind = \
            self._split_words(words)
        results = np.zeros((X.shape[0],
                            self._n_iterated_sums(words),
                            X.shape[2]))
        ext = (self.mode == "extended")
        for i, index in enumerate(simple_words_ind):
            l = len(simple_words[i]) if ext else 1
            results[:, index:index+l, :] = _fast_ISS(X,
                self._transform_simple_word(simple_words[i]),
                np.array([0] + simple_words[i].alpha + [0],
                         dtype=np.float32), ext)
        for i, index in enumerate(general_words_ind):
            l = len(general_words[i]) if ext else 1
            results[:, index:index+l, :] = _slow_ISS(X,
                general_words[i],
                np.array([0] + general_words[i].alpha + [0],
                         dtype=np.float32), ext)
        return results

    def copy(self) -> "ISSCalculator":
        """Returns a copy of this calculator.

        :rtype: ISSCalculator
        """
        calc = ISSCalculator(mode=self.mode)
        return calc


def ISS(X: np.ndarray,
        words: Union[List[Word], Word],
        mode: str = "whole") -> np.ndarray:
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
    :param mode: Mode of the used calculator. Has to be either "whole"
        or "extended".
    :type mode: str
    :returns: Numpy array of shape
        ``(X.shape[0], len(words), X.shape[2])``.
    :rtype: numpy.ndarray
    """
    words = [words] if isinstance(words, Word) else words

    calculator = ISSCalculator(mode)

    return calculator.calculate(X, words)
