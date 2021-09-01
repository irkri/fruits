from typing import List, Union

import numpy as np

from fruits.base.scope import force_input_shape, check_input_shape
from fruits.core.wording import Word, SimpleWord
from fruits.core.backend import _fast_ISS, _slow_ISS

class ISSCalculator:
    """Class that is responsible for managing the calculation of
    iterated sums.

    :param X: Time series dataset as numpy array of three dimensions.
    :type X: np.ndarray
    :param words: List of words used to calculate the iterated sums.
    :type words: List[Word]
    """
    def __init__(self,
                 X: np.ndarray,
                 words: List[Word]):
        self._X = X
        if not check_input_shape(X):
            self._X = force_input_shape(X)
        self._split_words(words)

    def _split_words(self, words: List[Word]):
        # splits the list of words into seperate ones for simple and 
        # complex words and saves the indices of their original
        # position in the given list
        if len(words) == 0:
            raise ValueError("At least one word for ISS calculation needed")
        self._simple_words = []
        self._simple_words_index = []
        self._complex_words = []
        self._complex_words_index = []
        for i, word in enumerate(words):
            if isinstance(word, SimpleWord):
                self._simple_words.append(word)
                self._simple_words_index.append(i)
            elif isinstance(word, Word):
                self._complex_words.append(word)
                self._complex_words_index.append(i)
            else:
                raise TypeError("Given words have to be objects of a "
                                "class that inherits from Word")

    def _transform_simple_word(self, word: SimpleWord) -> np.ndarray:
        # transforms all simplewords for faster calculation with a
        # backend function
        simple_word_raw = [el for el in word]
        word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
        for i in range(len(simple_word_raw)):
            for j in range(len(simple_word_raw[i])):
                word_transformed[i, j] = simple_word_raw[i][j]
        return word_transformed

    def calculate(self) -> np.ndarray:
        """Does the already initilialized calculation and returns the
        results.

        :rtype: np.ndarray
        """
        results = np.zeros((self._X.shape[0],
                            len(self._simple_words) + len(self._complex_words),
                            self._X.shape[2]))

        for i, index in enumerate(self._simple_words_index):
            results[:, index, :] = _fast_ISS(self._X,
                self._transform_simple_word(self._simple_words[i]),
                np.array([0] + self._simple_words[i].alpha + [0],
                         dtype=np.float32))
        for i, index in enumerate(self._complex_words_index):
            results[:, index, :] = _slow_ISS(self._X, self._complex_words[i],
                np.array([0] + self._complex_words[i].alpha + [0],
                         dtype=np.float32))
        return results


def ISS(X: np.ndarray,
        words: Union[List[Word], Word]) -> np.ndarray:
    """Takes in a number of time series and a list of words and
    calculates the iterated sums for each time series in ``X``.

    This function returns the iteratively calulcated cummulative sums of
    the input data, which will be stepwise transformed using the
    specified words.

    :param X: Three dimensional numpy array containing a
        multidimensional time series dataset.
    :type X: numpy.ndarray
    :type words: Union[List[Word], Word]
    :returns: Numpy array of shape
        ``(X.shape[0], len(words), X.shape[2])``.
    :rtype: numpy.ndarray
    """
    words = [words] if isinstance(words, Word) else words

    calculator = ISSCalculator(X, words)

    return calculator.calculate()
