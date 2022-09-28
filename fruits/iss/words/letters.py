__all__ = ["ExtendedLetter", "get_available", "letter"]

from functools import wraps
from typing import Callable, Optional, Union, overload

import numpy as np

BOUND_LETTER_TYPE = Callable[[np.ndarray], np.ndarray]
UNBOUND_LETTER_TYPE = Callable[[int], BOUND_LETTER_TYPE]


class ExtendedLetter:
    """Class for an extended letter used in words.
    A :class:`~fruits.words.word.Word` consists of a number of
    extended letters.
    An extended letter is a container that only allows contains
    functions that were decorated as a
    :meth:`~fruits.words.letters.letter`.

    Args:
        letter_string (str, optional): A string like
            ``f1(i)f2(j)f3(k)``, where ``f1,f2,f3`` are the names of
            decorated letters and ``i,j,k`` are integers representing
            dimensions. For available letters call
            :meth:`fruits.words.letters.get_available`.
    """

    def __init__(self, letter_string: str = "") -> None:
        self._letters: list[UNBOUND_LETTER_TYPE] = []
        self._dimensions: list[int] = []
        self._string_repr = ""
        self._iter_index = -1
        self._append_from_string(letter_string)

    def _append_from_string(self, letter_string: str) -> None:
        letters = letter_string.split(")")[:-1]
        for letter in letters:
            l, d = letter.split("(")
            self.append(l, int(d)-1)

    def append(self, letter: str, dim: int = 0) -> None:
        """Appends a letter to the ExtendedLetter.

        Args:
            letter (callable): Function that was decorated with
                :meth:`~fruits.words.letters.letter`.
            dim (int): Dimension of the letter that is going to be used
                as its second argument, if it has one. Defaults to 0.
        """
        self._letters.append(_get(letter))
        self._dimensions.append(dim)
        self._string_repr += letter
        self._string_repr += "(" + str(dim+1) + ")"

    def copy(self) -> "ExtendedLetter":
        """Returns a copy of this extended letter."""
        el = ExtendedLetter()
        el._letters = self._letters.copy()
        el._dimensions = self._dimensions.copy()
        el._string_repr = self._string_repr
        return el

    def __iter__(self) -> "ExtendedLetter":
        self._iter_index = -1
        return self

    def __next__(self) -> BOUND_LETTER_TYPE:
        if self._iter_index < len(self._letters)-1:
            self._iter_index += 1
            return self._letters[self._iter_index](
                        self._dimensions[self._iter_index])
        raise StopIteration()

    def __len__(self) -> int:
        return len(self._letters)

    def __getitem__(self, i: int) -> BOUND_LETTER_TYPE:
        return self._letters[i](self._dimensions[i])

    def __str__(self) -> str:
        return "[" + self._string_repr + "]"


def simple(i: int) -> BOUND_LETTER_TYPE:
    def index_manipulation(X: np.ndarray) -> np.ndarray:
        return X[i, :]
    return index_manipulation


def absolute(i: int) -> BOUND_LETTER_TYPE:
    def index_manipulation(X: np.ndarray) -> np.ndarray:
        return np.abs(X[i, :])
    return index_manipulation


_AVAILABLE: dict[str, UNBOUND_LETTER_TYPE] = {
    "DIM": simple,
    "ABS": absolute,
}


def _log(name: str, func: UNBOUND_LETTER_TYPE) -> None:
    if name in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' already exists")
    _AVAILABLE[name] = func


def _get(name: str) -> UNBOUND_LETTER_TYPE:
    # returns the corresponding letter for the given name
    if name not in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' does not exist")
    return _AVAILABLE[name]


def get_available() -> list[str]:
    """Returns a list of all available letter names to use in a
    :class:`~fruits.words.letters.ExtendedLetter`.
    """
    return list(_AVAILABLE.keys())


@overload
def letter(*args, name: None = None) -> UNBOUND_LETTER_TYPE:
    ...


@overload
def letter(*args, name: str = "") -> Callable[..., UNBOUND_LETTER_TYPE]:
    ...


def letter(
    *args,
    name: Optional[str] = None,
) -> Union[UNBOUND_LETTER_TYPE, Callable[..., UNBOUND_LETTER_TYPE]]:
    """Decorator for the implementation of a letter appendable to an
    :class:`~fruits.words.letters.ExtendedLetter`.

    It is possible to implement a new letter by using this decorator.
    This callable (e.g. called ``myletter``) has to have two arguments:
    ``X: np.ndarray`` and ``i: int``, where ``X`` is a multidimensional
    time series and ``i`` is the dimension index that can
    (but doesn't need to) be used in the decorated function.
    The function has to return a numpy array. ``X`` has exactly two
    dimensions and the returned array has one dimension.

    .. code-block:: python

        @fruits.words.letter(name="ReLU")
        def myletter(X: np.ndarray, i: int) -> np.ndarray:
            return X[i, :] * (X[i, :]>0)

    It is also possible to use this decorator without any arguments. The
    following code does the same thing as the above example.

    .. code-block:: python

        @fruits.words.letter
        def ReLU(X: np.ndarray, i: int) -> np.ndarray:
            return X[i, :] * (X[i, :]>0)

    Available predefined letters are:

        - ``DIM``: Extracts a single dimension.
        - ``ABS``: Extracts the absolute value of one dimension.

    Args:
        name (str, optional): You can supply a name to the function.
            This name will be used for documentation in an
            ``ExtendedLetter`` object. If no name is supplied, then the
            name of the function is used. Each letter has to have a
            unique name. They are later used to refer to the specific
            functions in word creation.
    """
    if len(args) > 1:
        raise RuntimeError("Too many arguments")

    if name is None and len(args) == 1 and callable(args[0]):

        @wraps(args[0])
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return args[0](X, i)
            return index_manipulation
        _log(args[0].__name__, wrapper)

        return wrapper

    if name is None:
        raise ValueError(
            "Please either specify the 'name' argument or use this "
            "decorator without calling it."
        )

    def letter_decorator(func):

        @wraps(func)
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return func(X, i)
            return index_manipulation
        _log(name, wrapper)

        return wrapper

    return letter_decorator
