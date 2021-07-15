from functools import wraps

import numpy as np

COMPLEX_LETTER_SIGNATURE = "fruits_letter"
COMPLEX_LETTER_NAME = "fruits_name"

class ExtendedLetter:
    """Class for an extended letter used in ``fruits.core.ComplexWord``.
    
    An ExtendedLetter object is a container that only allows
    appending functions that were decorated with
    ``fruits.core.complex_letter``.
    It can then be used in a word to calculate iterated sums of a
    multidimensional time series.

    :param *letters: Letters to add to the object. One letter is given
        as a tuple ``(letter, dimension)``, where ``letter`` is the
        function decorated with the ``complex_letter`` decorator and
        ``dimension`` is the second argument of ``letter``.
    :type *letters: One or more tuples
    """
    def __init__(self, *letters):
        self._letters = []
        self._dimensions = []
        self._string_repr = ""
        for letter in letters:
            if not isinstance(letter, tuple) or len(letter) != 2:
                raise TypeError("ExtendedLetter can only be initialized " +
                                "using tupels")
            self.append(*letter)

    def append(self, letter: callable, dim: int = 0):
        """Appends a letter to the ExtendedLetter object.
        
        :param letter: Function that was decorated with
            ``fruits.core.complex_letter``.
        :type letter: callable
        :param int: Dimension of the letter that is going to be used as
            its second argument, if it has one., defaults to 0
        :type dim: int, optional
        """
        if not callable(letter):
            raise TypeError("Argument letter has to be a callable function")
        elif not _letter_configured(letter):
            raise TypeError("Letter has the wrong signature. Perhaps it " +
                            "wasn't decorated correctly?")
        else:
            self._letters.append(letter)
            self._dimensions.append(dim)
            self._string_repr += letter.__dict__[COMPLEX_LETTER_NAME]
            self._string_repr += "(" + str(dim+1) + ")"

    def copy(self):
        """Returns a copy of this extended letter.
        
        :rtype: ExtendedLetter
        """
        el = ExtendedLetter()
        el._letters = self._letters.copy()
        el._dimensions = self._dimensions.copy()
        return el

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        if self._iter_index < len(self._letters)-1:
            self._iter_index += 1
            return self._letters[self._iter_index](
                        self._dimensions[self._iter_index])
        raise StopIteration()

    def __len__(self) -> int:
        return len(self._letters)

    def __getitem__(self, i: int) -> callable:
        return self._letters[i](self._dimensions[i])

    def __copy__(self):
        return self.copy()

    def __str__(self) -> str:
        return "["+self._string_repr+"]"

    def __repr__(self):
        return "fruits.core.letters.ExtendedLetter"


def complex_letter(*args, name: str = None):
    """Decorator for the implementation of a complex letter appendable
    to an ``ExtendedLetter`` object.
    
    It is possible to implement your own complex letters by using this
    decorator. Your complex letter has to be a function (e.g. called
    'myletter') that has two arguments: ``X: np.ndarray`` and
    ``i: int``, where ``X`` is a multidimensional time series and ``i``
    is the dimension index that can (but doesn't need to) be used in the
    decorated function.
    The function has to return a numpy array. ``X`` has exactly two
    dimensions and the returned array has one dimension.

    .. code-block:: python

        @fruits.core.complex_letter(name="ReLU")
        def myletter(X: np.ndarray, i: int) -> np.ndarray:
            return X[i, :] * (X[i, :]>0)

    It is also possible to use this decorator without any arguments:

    .. code-block:: python

        @fruits.core.complex_letter

    :param name: You can supply a name to the function. This name will
        be used for documentation in an ``ExtendedLetter`` object. If
        no name is supplied, then the name of the function is used.,
        defaults to None
    :type name: str, optional
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("Unknown argument type for name")
    if len(args) > 1:
        raise RuntimeError("Too many arguments")
    if name is None and len(args)==1 and callable(args[0]):
        _configure_letter(args[0])
        @wraps(args[0])
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return args[0](X, i)
            return index_manipulation
        return wrapper
    else:
        if name is None and len(args) > 0:
            if not isinstance(args[0], str):
                raise TypeError("Unknown argument type")
            name = args[0]
        def complex_letter_decorator(func):
            _configure_letter(func, name=name)
            @wraps(func)
            def wrapper(i: int):
                def index_manipulation(X: np.ndarray):
                    return func(X, i)
                return index_manipulation
            return wrapper
        return complex_letter_decorator

def _configure_letter(func: callable, name: str = None):
    if func.__code__.co_argcount != 2:
        raise RuntimeError("Wrong number of arguments at decorated function " +
                           str(func.__name__) + ". Should be 2.")
    func.__dict__[COMPLEX_LETTER_SIGNATURE] = "complex_letter"
    if name is None:
        func.__dict__[COMPLEX_LETTER_NAME] = func.__name__
    else:
        func.__dict__[COMPLEX_LETTER_NAME] = name

def _letter_configured(func: callable) -> bool:
    if (COMPLEX_LETTER_SIGNATURE in func.__dict__ and
        COMPLEX_LETTER_NAME in func.__dict__):
        return True
    return False

@complex_letter(name="simple_letter")
def simple_letter(X: np.ndarray, i: int) -> np.ndarray:
    return X[i, :]
