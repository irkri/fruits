"""This small python module is an appendix to the package fruits and
allows the creation of more complex SummationIterator objects where
the letters are arbitrary methods that work on multidimensional time
series.
"""

import numpy as np

from context import fruits

# each complex letter is defined by making use of the complex_letter
# decorator

def complex_letter(name: str):
    def complex_letter_decorator(func):
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return func(X, i)
            return index_manipulation
        wrapper.__doc__ = name
        return wrapper
    return complex_letter_decorator

# complex letters implementation

@complex_letter(name="SIGMOID")
def sigmoid(X: np.ndarray, i: int):
    return 1 / (1 + np.exp(-X[i, :]))

@complex_letter(name="leakyRELU")
def leaky_relu(X: np.ndarray, i: int):
    out = np.zeros(X.shape[1], dtype=np.float64)
    out += X[i, :] * (X[i, :]>0)
    out += (X[i, :]*0.005) * (X[i, :]<=0)
    return out

@complex_letter(name="TANH")
def tanh(X: np.ndarray, i: int):
    pos = np.exp(0.001*X[i, :])
    neg = np.exp(-0.001*X[i, :])
    return (pos-neg) / (pos+neg)

@complex_letter(name="ID")
def id_(X: np.ndarray, i: int):
    return X[i, :]

# convenience functions for creation of complex SummationIterators

def generate_complex_words(simple_words, letter, scale: int = 0):
    """Generate complex complex words by replacing letters of a simple
    word with a given function.
    
    :param simple_words: SimpleWord objects to replace.
    :type simple_words: list
    :param letter: Function that replaces the letters in all
        SimpleWords
    :type letter: callable, decorated with complex_letter
    :param scale: Scale for all SummationIterator objects, defaults to 0
    :type scale: int, optional
    :returns: List of SummationIterator objects
    :rtype: list
    """
    complex_words = []
    for simple_word in simple_words:
        complex_words.append(fruits.iterators.SummationIterator(
                             str(simple_word)[11:-1]))
        for monomial in simple_word.monomials():
            mon = []
            for i, letter_ in enumerate(monomial):
                for l in range(letter_):
                    mon.append(letter(i))
        complex_words[-1].multiply(mon)
        complex_words[-1].scale = scale
    return complex_words

def generate_random_complex_words(simple_words,
                                  letters: list = [
                                                   sigmoid,
                                                   leaky_relu,
                                                   tanh,
                                                   id_,
                                                  ],
                                  scale: int = 0):
    """Generate random complex words. Every letter in each SimpleWord
    is replaced by a random complex letter.
    
    :param simple_words: list of SimpleWord objects
    :type simple_words: list
    :param letters: List of complex letters to choose randomly from.,
        defaults to [sigmoid, leaky_relu, tanh, id_]
    :type letters: list
    :param scale: Scale of the resulting SummationIterators,
        defaults to 0
    :type scale: int, optional
    :returns: List of SummationIterator objects
    :rtype: list
    """
    complex_words = []
    for simple_word in simple_words:
        complex_words.append(fruits.iterators.SummationIterator())
        new_name = simple_word.name
        new_name_index = 0
        for monomial in simple_word.monomials():
            new_name_index += 1
            mon = []
            for i, letter in enumerate(monomial):
                for l in range(letter):
                    new_name = new_name[:new_name_index] + \
                               new_name[new_name_index+1:]
                    k = np.random.randint(0, len(letters), 1)[0]
                    new_name = new_name[:new_name_index] + \
                               letters[k].__doc__+f"({i+1})" + \
                               new_name[new_name_index:]
                    mon.append(letters[k](i))
                    new_name_index += len(letters[k].__doc__+f"({i+1})")
                    start = False
            new_name_index += 1
        complex_words[-1].name = new_name
        complex_words[-1].multiply(mon)
        complex_words[-1].scale = scale
    return complex_words

def generate_rotated_complex_words(simple_words,
                                   letters: list = [
                                                    id_,
                                                    sigmoid,
                                                    leaky_relu,
                                                    tanh,
                                                   ],
                                   scale: int = 0):
    """Generate complex SummationIterator objects by replacing letters
    in each SimpleWord with complex letters choosen of-after-another
    from the list 'letters' (if end is reached, start from beginning).
    
    :param simple_words: List of SimpleWord objects
    :type simple_words: list
    :param letters: List of complex words.,
        defaults to [id_, sigmoid, leaky_relu, tanh]
    :type letters: list, optional
    :param scale: Scale of the resulting SummationIterator objects.,
        defaults to 0
    :type scale: int, optional
    :returns: List of complex words.
    :rtype: list
    """
    complex_words = []
    function_index = 0
    for simple_word in simple_words:
        complex_words.append(fruits.iterators.SummationIterator())
        new_name = simple_word.name
        new_name_index = 0
        for monomial in simple_word.monomials():
            new_name_index += 1
            mon = []
            for i, letter in enumerate(monomial):
                for l in range(letter):
                    new_name = new_name[:new_name_index] + \
                               new_name[new_name_index+1:]
                    function_index = (function_index + 1) % len(letters)
                    new_name = new_name[:new_name_index] + \
                               letters[function_index].__doc__+f"({i+1})" + \
                               new_name[new_name_index:]
                    mon.append(letters[function_index](i))
                    new_name_index += len(letters[function_index].__doc__+
                                          f"({i+1})")
            new_name_index += 1
        complex_words[-1].name = new_name
        complex_words[-1].multiply(mon)
        complex_words[-1].scale = scale
    return complex_words
