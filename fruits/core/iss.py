import numba
import numpy as np

from fruits.core.wording import AbstractWord, SimpleWord

def ISS(Z: np.ndarray, words: list) -> np.ndarray:
    """Takes in a number of time series and a list of AbstractWord
    objects and calculated the iterated sums for each time series in
    ``Z``.

    For each given time series ``Z``, this function returns the
    iteratively calulcated cummulative sums of the input data, which
    will be stepwise transformed using the specified AbstractWord[s].

    :param Z: Three dimensional numpy array containing a
        multidimensional time series dataset.
    :type Z: numpy.ndarray
    :param words: List of AbstractWord objects or a single AbstractWord.
    :type words: list or AbstractWord
    :returns: Numpy array of shape
        ``(Z.shape[0], len(words), Z.shape[2])``.
    :rtype: numpy.ndarray
    """
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
                                    max_dim + 1))
        for i in range(len(simple_words_raw)):
            for j in range(len(simple_words_raw[i])):
                for k in range(len(simple_words_raw[i][j])):
                    simple_words_tf[i, j, k] = simple_words_raw[i][j][k]
        ISS_fast = _fast_ISS(Z, simple_words_tf)

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

def _slow_ISS(Z: np.ndarray,
              words: list) -> np.ndarray:
    # calculates the iterated sums for Z and a given list of
    # general ComplexWords
    result = np.zeros((Z.shape[0], len(words), Z.shape[2]))
    for i in range(Z.shape[0]):
        for j in range(len(words)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            for k, el in enumerate(words[j]):
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(el)):
                    C = C * el[l](Z[i, :, :])
                result[i, j, :] = np.cumsum(result[i, j, :] * C)

    return result

@numba.njit(parallel=True, fastmath=True)
def _fast_ISS(Z: np.ndarray, 
              words: np.ndarray) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], len(words), Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        for j in numba.prange(len(words)):
            result[i, j, :] = np.ones(Z.shape[2], dtype=np.float64)
            for k in range(len(words[j])):
                if not np.any(words[j][k]):
                    continue
                C = np.ones(Z.shape[2], dtype=np.float64)
                for l in range(len(words[j][k])):
                    if words[j][k][l] != 0:
                        C = C * Z[i, l, :]**words[j][k][l]
                result[i, j, :] = np.cumsum(result[i, j, :] * C)
    return result
