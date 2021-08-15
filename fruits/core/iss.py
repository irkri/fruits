from typing import List, Union

import numpy as np

from fruits.base.scope import force_input_shape, check_input_shape
from fruits.core.wording import AbstractWord, SimpleWord
from fruits.core.backend import _fast_ISS, _slow_ISS

class ISSCalculator:
    """Class that is responsible for managing the calculation of
    iterated sums.

    :param X: Time series dataset as numpy array of three dimensions.
    :type X: np.ndarray
    :param words: List of words used to calculate the iterated sums.
    :type words: List[AbstractWord]
    """
    def __init__(self, X: np.ndarray, words: List[AbstractWord]):
        self._X = X
        if not check_input_shape(X):
            self._X = force_input_shape(X)
        self._split_words(words)

    def _split_words(self, words: List[AbstractWord]):
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
            elif isinstance(word, AbstractWord):
                self._complex_words.append(word)
                self._complex_words_index.append(i)
            else:
                raise TypeError("Given words have to be objects of a "
                                "class that inherits from AbstractWord")

    def _transform_simple_word(self, word: SimpleWord):
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
                self._transform_simple_word(self._simple_words[i]))
        for i, index in enumerate(self._complex_words_index):
            results[:, index, :] = _slow_ISS(self._X, self._complex_words[i])

        return results


def ISS(X: np.ndarray,
        words: Union[List[AbstractWord], AbstractWord]) -> np.ndarray:
    """Takes in a number of time series and a list of words and
    calculates the iterated sums for each time series in ``X``.

    This function returns the iteratively calulcated cummulative sums of
    the input data, which will be stepwise transformed using the
    specified words.

    :param X: Three dimensional numpy array containing a
        multidimensional time series dataset.
    :type X: numpy.ndarray
    :param words: List of AbstractWord objects or a single AbstractWord.
    :type words: List[AbstractWord] or AbstractWord
    :returns: Numpy array of shape
        ``(X.shape[0], len(words), X.shape[2])``.
    :rtype: numpy.ndarray
    """
<<<<<<< Updated upstream
    if isinstance(words, AbstractWord):
        words = [words]

    # divide words into seperate lists for SimpleWords and ComplexWords
    simple_words = []
    simple_words_index = []
    complex_words = []
    complex_words_index = []
    for i, word in enumerate(words):
        if isinstance(word, SimpleWord):
            simple_words.append(word)
            simple_words_index.append(i)
        elif isinstance(word, AbstractWord):
            complex_words.append(word)
            complex_words_index.append(i)
        else:
            raise TypeError("Given words have to be objects of a class " +
                            "that inherits from AbstractWord")

    # get solution for the SimpleWords
    # transform each SimpleWord in a way such that every word has the
    # same number of extended letters and each extended letter has the
    # same number of letters
    if simple_words:
        max_dim = max(word._max_dim for word in simple_words)
        simple_words_raw = [[el for el in word] for word in simple_words]
        max_word_length = max(len(els) for els in simple_words_raw)
        simple_words_tf = np.zeros((len(simple_words), 
                                    max_word_length,
                                    max_dim + 1), dtype=np.int32)
        for i in range(len(simple_words_raw)):
            for j in range(len(simple_words_raw[i])):
                for k in range(len(simple_words_raw[i][j])):
                    simple_words_tf[i, j, k] = simple_words_raw[i][j][k]
        ISS_fast = _fast_ISS(Z.astype(np.float64), simple_words_tf)

    # get solution for AbstractWords that are not of type SimpleWord
    if complex_words:
        ISS_slow = _slow_ISS(Z, complex_words)

    # concatenate results if both types of words were specified
    if simple_words and complex_words:
        results = np.zeros((Z.shape[0], len(words), Z.shape[2]))
        for i, index in enumerate(simple_words_index):
            results[:, index, :] = ISS_fast[:, i, :]
        for i, index in enumerate(complex_words_index):
            results[:, index, :] = ISS_slow[:, i, :]
        return results
    elif simple_words:
        return ISS_fast
    elif complex_words:
        return ISS_slow

def _slow_ISS(Z: np.ndarray, words: list) -> np.ndarray:
    # calculates the iterated sums for Z and a given list of
    # general ComplexWords
    result = np.zeros((Z.shape[0], len(words), Z.shape[2]))
    for i in range(Z.shape[0]):
        for j in range(len(words)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            r = len(words[j])
            for k, el in enumerate(words[j]):
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(el)):
                    C = C * el[l](Z[i, :, :])
                result[i, j, :] = _fast_CS(result[i, j, :] * C,
                                           int(bool(r-(k+1))))
    return result

@numba.njit("float64[:](float64[:], int32)", cache=True)
def _fast_CS(Z: np.ndarray, r: int):
    Z = np.roll(np.cumsum(Z), r)
    Z[:r] = 0
    return Z

@numba.njit("float64[:](float64[:,:], int32[:,:])",
            fastmath=True, cache=True)
def _fast_single_ISS(Z: np.ndarray, word: np.ndarray) -> np.ndarray:
    result = np.ones(Z.shape[1], dtype=np.float64)
    r = len(word)
    for k in range(r):
        if not np.any(word[k]):
            continue
        C = np.ones(Z.shape[1], dtype=np.float64)
        for l in range(len(word[k])):
            if word[k][l] != 0:
                C = C * Z[l, :]**word[k][l]
        result = _fast_CS(result * C, int(bool(r-(k+1))))
    return result

@numba.njit("float64[:,:,:](float64[:,:,:], int32[:,:,:])",
            parallel=True, cache=True)
def _fast_ISS(Z: np.ndarray, words: np.ndarray) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], len(words), Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        for j in numba.prange(len(words)):
            result[i, j, :] = _fast_single_ISS(Z[i, :, :], words[j])
    return result
=======
    words = [words] if isinstance(words, AbstractWord) else words

    calculator = ISSCalculator(X, words)

    return calculator.calculate()
>>>>>>> Stashed changes
