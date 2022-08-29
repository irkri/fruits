from functools import partial, wraps
from typing import Callable, Optional, Union, overload

import numpy as np

LETTER_NAME = "fruits_letter_name"

BOUND_LETTER_TYPE = Callable[[np.ndarray], np.ndarray]
FREE_LETTER_TYPE = Callable[[int], BOUND_LETTER_TYPE]


class ExtendedLetter:
    """Class for an extended letter used in words.
    A :class:`~fruits.words.word.Word` consists of a number of
    extended letters.
    An extended letter is a container that only allows appending
    functions that were decorated with
    :meth:`~fruits.words.letters.letter`.

    Args:
        letter_string (str, optional): A string like
            ``f1(i)f2(j)f3(k)``, where ``f1,f2,f3`` are the names of
            decorated letters and ``i,j,k`` are integers representing
            dimensions. For available letters call
            :meth:`fruits.words.letters.get_available`.
    """

    def __init__(self, letter_string: str = ""):
        self._letters: list[FREE_LETTER_TYPE] = []
        self._dimensions: list[int] = []
        self._string_repr = ""
        self.append_from_string(letter_string)

    def append(self, letter: FREE_LETTER_TYPE, dim: int = 0) -> None:
        """Appends a letter to the ExtendedLetter object.

        Args:
            letter (callable): Function that was decorated with
                :meth:`~fruits.words.letters.letter`.
            dim (int): Dimension of the letter that is going to be used
                as its second argument, if it has one. Defaults to 0.
        """
        if not callable(letter):
            raise TypeError("Argument letter has to be a callable function")
        elif not _is_letter(letter):
            raise TypeError("Letter has the wrong signature. Perhaps it " +
                            "wasn't decorated correctly?")
        else:
            self._letters.append(letter)
            self._dimensions.append(dim)
            self._string_repr += letter.__dict__[LETTER_NAME]
            self._string_repr += "(" + str(dim+1) + ")"

    def append_from_string(self, letter_string: str) -> None:
        letters = letter_string.split(")")[:-1]
        for letter in letters:
            l, d = letter.split("(")
            self.append(_get(l), int(d)-1)

    def copy(self) -> "ExtendedLetter":
        el = ExtendedLetter()
        el._letters = self._letters.copy()
        el._dimensions = self._dimensions.copy()
        el._string_repr = self._string_repr
        return el

    def __iter__(self) -> "ExtendedLetter":
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

    def __str__(self) -> str:
        return "[" + self._string_repr + "]"


@overload
def letter(*args, name: None = None) -> FREE_LETTER_TYPE:
    ...


@overload
def letter(*args, name: str = "") -> Callable[..., FREE_LETTER_TYPE]:
    ...


def letter(
    *args,
    name: Optional[str] = None,
) -> Union[FREE_LETTER_TYPE, Callable[..., FREE_LETTER_TYPE]]:
    """Decorator for the implementation of a letter appendable to an
    :class:`~fruits.words.letters.ExtendedLetter` object.

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

    It is also possible to use this decorator without any arguments:

    .. code-block:: python

        @fruits.words.letter

    Available predefined letters are:

        - ``simple``: Extracts a single dimension
        - ``absolute``: Extracts the absolute value of a single dim.

    Args:
        name (str, optional): You can supply a name to the function.
            This name will be used for documentation in an
            ``ExtendedLetter`` object. If no name is supplied, then the
            name of the function is used. Each letter has to have a
            unique name.
    """
    if len(args) > 1:
        raise RuntimeError("Too many arguments")
    if name is None and len(args) == 1 and callable(args[0]):
        _configure_letter(args[0], args[0].__name__)

        @wraps(args[0])
        def wrapper(i: int):
            def index_manipulation(X: np.ndarray):
                return args[0](X, i)
            return index_manipulation
        _log(args[0].__name__, wrapper)

        return wrapper
    else:
        if name is None:
            raise ValueError(
                "Please either specify the 'name' argument or use this "
                "decorator without calling it."
            )

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


def _log(name: str, func: FREE_LETTER_TYPE) -> None:
    if name in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' already exists")
    _AVAILABLE[name] = func


def _get(name: str) -> FREE_LETTER_TYPE:
    # returns the corresponding letter for the given name
    if name not in _AVAILABLE:
        raise RuntimeError(f"Letter with name '{name}' does not exist")
    return _AVAILABLE[name]


def get_available() -> list[str]:
    """Returns a list of all available letter names to use in a
    :class:`~fruits.words.letters.ExtendedLetter`.
    """
    return list(_AVAILABLE.keys())


def _configure_letter(func: BOUND_LETTER_TYPE, name: str) -> None:
    # marks the input callable as a letter
    if func.__code__.co_argcount != 2:
        raise RuntimeError(
            "Wrong function signature for letter configuration. "
            "Should be 'letter(X: numpy.ndarray, i: int)'."
        )
    func.__dict__[LETTER_NAME] = name


def _is_letter(func: FREE_LETTER_TYPE) -> bool:
    # checks if the given callable is a letter
    if (LETTER_NAME in func.__dict__
            and func.__dict__[LETTER_NAME] in _AVAILABLE):
        return True
    return False


@letter(name="SIMPLE")
def simple(X: np.ndarray, i: int) -> np.ndarray:
    return X[i, :]


@letter(name="ABS")
def absolute(X: np.ndarray, i: int) -> np.ndarray:
    return np.abs(X[i, :])
