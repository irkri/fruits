"""This small python module is an appendix to the package FRUITS and
allows more ways for constructing ComplexWord objects.
"""

import numpy as np

from context import fruits

# complex letters implementation

@fruits.core.complex_letter(name="SIGMOID")
def sigmoid(X: np.ndarray, i: int):
    return 1 / (1 + np.exp(-0.001*X[i, :]))

@fruits.core.complex_letter(name="leakyRELU")
def leaky_relu(X: np.ndarray, i: int):
    out = np.zeros(X.shape[1], dtype=np.float64)
    out += X[i, :] * (X[i, :]>0)
    out += (X[i, :]*0.005) * (X[i, :]<=0)
    return out

@fruits.core.complex_letter(name="TANH")
def tanh(X: np.ndarray, i: int):
    pos = np.exp(0.001*X[i, :])
    neg = np.exp(-0.001*X[i, :])
    return (pos-neg) / (pos+neg)

@fruits.core.complex_letter(name="ID")
def id_(X: np.ndarray, i: int):
    return X[i, :]

def simplewords_replace_letters_randomly(simple_words,
                                         letters: list):
    """Generates random complex words. Every letter in each SimpleWord
    is replaced by a random complex letter in the given list.
    
    :param simple_words: list of SimpleWord objects
    :type simple_words: list
    :param letters: List of complex letters to choose randomly from.
    :type letters: list
    :returns: List of ComplexWord objects
    :rtype: list
    """
    complex_words = []
    for simple_word in simple_words:
        complex_words.append(fruits.core.ComplexWord())
        for el in simple_word:
            ext_letter = fruits.core.ExtendedLetter()
            for i, dim in enumerate(el):
                for l in range(dim):
                    k = np.random.randint(0, len(letters))
                    ext_letter.append(letters[k], i)
            complex_words[-1].multiply(ext_letter)
    return complex_words

def simplewords_replace_letters_sequentially(simple_words,
                                             letters: list):
    """Generates complex words. Every letter in each SimpleWord
    is replaced by another letter sequentially chosen from the given
    list.
    
    :param simple_words: list of SimpleWord objects
    :type simple_words: list
    :param letters: List of complex letters to iterate through.
    :type letters: list
    :returns: List of ComplexWord objects
    :rtype: list
    """
    complex_words = []
    k = 0
    for simple_word in simple_words:
        complex_words.append(fruits.core.ComplexWord())
        for el in simple_word:
            ext_letter = fruits.core.ExtendedLetter()
            for i, dim in enumerate(el):
                for l in range(dim):
                    ext_letter.append(letters[k], i)
                    k = (k+1) % len(letters)
            complex_words[-1].multiply(ext_letter)
    return complex_words
