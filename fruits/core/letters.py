from functools import wraps
from typing import List, Tuple, Callable

import numpy as np

LETTER_SIGNATURE = "fruits_letter"
LETTER_NAME = "fruits_name"

BOUND_LETTER_TYPE = Callable[[np.ndarray, int], np.ndarray]
FREE_LETTER_TYPE = Callable[[int], BOUND_LETTER_TYPE]

class ExtendedLetter:
    """Class for an extended letter used in words.
    A :class:`~fruits.core.wording.Word` consists of a number of
    extended letters.
    An extended letter is a container that only allows appending
    functions that were decorated with
    :meth:`~fruits.core.letters.letter`.

    :param letter_string: A string like ``f1(i)f2(j)f3(k)``, where
        ``f1,f2,f3`` are the names of decorated letters and ``i,j,k``
        are integers representing dimensions. For available letters call
        :meth:`fruits.core.letters.get_available`.
    :type letter_string: str
    """
    def __init__(self, letter_string: str = ""):
        self._letters = []
        self._dimensions = []
        self._string_repr = ""
        self.append_from_string(letter_string)

    def append(self, letter: FREE_LETTER_TYPE, dim: int = 0):
        """Appends a letter to the ExtendedLetter object.

        :param letter: Function that was decorated with
            :meth:`~fruits.core.letters.letter`.
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
            self._string_repr += letter.__dict__[LETTER_NAME]
            self._string_repr += "(" + str(dim+1) + ")"

    def append_from_string(self, letter_string: str):
        letters = letter_string.split(")")[:-1]
        for letter in letters:
            l, d = letter.split("(")
            self.append(_get(l), int(d))

    def copy(self) -> "ExtendedLetter":
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

    def __getitem__(self, i: int) -> BOUND_LETTER_TYPE:
        return self._letters[i](self._dimensions[i])

    def __copy__(self) -> "ExtendedLetter":
        return self.copy()

    def __str__(self) -> str:
        return "["+self._string_repr+"]"

    def __repr__(self):
        return "fruits.core.letters.ExtendedLetter"


def letter(*args, name: str = None):
    """Decorator for the implementation of a letter appendable to an
    :class:`fruits.core.letters.ExtendedLetter` object.

    It is possible to implement a new letter by using this decorator.
    This callable (e.g. called ``myletter``) has to have two arguments:
    ``X: np.ndarray`` and ``i: int``, where ``X`` is a multidimensional
    time series and ``i`` is the dimension index that can
    (but doesn't need to) be used in the decorated function.
    The function has to return a numpy array. ``X`` has exactly two
    dimensions and the returned array has one dimension.

    .. code-block:: python

        @fruits.core.letter(name="ReLU")
        def myletter(X: np.ndarray, i: int) -> np.ndarray:
            return X[i, :] * (X[i, :]>0)

    It is also possible to use this decorator without any arguments:

    .. code-block:: python

        @fruits.core.letter

    Available predefined letters are:

        - ``simple``: Extracts a single dimension
        - ``absolute``: Extracts the absolute value of a single dim.

    :param name: You can supply a name to the function. This name will
        be used for documentation in an ``ExtendedLetter`` object. If
        no name is supplied, then the name of the function is used.
        Each letter has to have a unique name., defaults to None
    :type name: str, optional
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("Unknown argument type for name")
    if len(args) > 1:
        raise RuntimeError("Too many arguments")
    if name is None and len(args)==1 and callable(args[0]):
        _configure_letter(args[0], args[0].__name__)
        @wraps(args[0])
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return args[0](X, i)
            return index_manipulation
        _log(args[0].__name__, wrapper)
        return wrapper
    else:
        if name is None and len(args) > 0:
            if not isinstance(args[0], str):
                raise TypeError("Unknown argument type")
            name = args[0]
        def letter_decorator(func):
            _configure_letter(func, name=name)
            @wraps(func)
            def wrapper(i: int):
                def index_manipulation(X: np.ndarray):
                    return func(X, i)
                return index_manipulation
            _log(name, wrapper)
            return wrapper
        return letter_decorator

_AVAILABLE = dict()

def _log(name: str, func: FREE_LETTER_TYPE):
    if name in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' already exists")
    _AVAILABLE[name] = func

def _get(name: str) -> FREE_LETTER_TYPE:
    # returns the corresponding letter for the given name
    if not name in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' does not exist")
    return _AVAILABLE[name]

def get_available() -> List[str]:
    """Returns a list of all available letter names to use in a
    :class:`fruits.core.letters.ExtendedLetter`.

    :rtype: List[str]
    """
    return list(_AVAILABLE.keys())

def _configure_letter(func: BOUND_LETTER_TYPE, name: str):
    # marks the input callable as a letter
    if func.__code__.co_argcount != 2:
        raise RuntimeError("Wrong number of arguments at decorated function " +
                           str(func.__name__) + ". Should be 2.")
    func.__dict__[LETTER_SIGNATURE] = "letter"
    func.__dict__[LETTER_NAME] = name

def _letter_configured(func: BOUND_LETTER_TYPE) -> bool:
    # checks if the given callable is a letter
    if (LETTER_SIGNATURE in func.__dict__ and
        LETTER_NAME in func.__dict__ and
        func.__dict__[LETTER_NAME] in _AVAILABLE):
        return True
    return False

@letter(name="SIMPLE")
def simple(X: np.ndarray, i: int) -> np.ndarray:
    return X[i, :]

@letter(name="ABS")
def absolute(X: np.ndarray, i: int) -> np.ndarray:
    return np.abs(X[i, :])
