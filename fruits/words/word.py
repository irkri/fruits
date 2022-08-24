import re
from typing import Optional, Union

from fruits.words.letters import ExtendedLetter


class Word:
    """A word is a collection of
    :class:`~fruits.words.letter.ExtendedLetter` objects.
    An extended letter is a collection of letters.
    A letter is a function that accepts and returns numpy arrays. The
    order of the letters in an extended letter doesn't matter.

    A Word is used for the calculation of iterated sums for a given
    time series dataset ``X``. This can be done by calling::

        fruits.signature.iss.ISS(X, word)

    Each Word is iterable. The items returned by the iterator are the
    extended letters.

    One can extend an already created word by calling
    ``word.multiply(...)``.

    Example:

    .. code-block:: python

        word = Word()

        el01 = ExtendedLetter()
        el01.append(fruits.words.letters.simple, 0)
        el01.append(fruits.words.letters.simple, 0)
        el02 = ExtendedLetter()
        el02.append(fruits.words.letters.simple, 0)
        el02.append(fruits.words.letters.simple, 1)
        el02.append(fruits.words.letters.simple, 1)

        word.multiply(el01)
        word.multiply(el02)

        iterated_sums = fruits.signature.ISS(X, word)

    The result in ``iterated_sums`` is (roughly) equal to

    .. code-block:: python

        numpy.cumsum(numpy.cumsum(X[0, :]**2) * X[0, :]*X[1, :]**2)

    which can be simplified using a
    :class:`~fruits.words.word.SimpleWord`: ::

        fruits.signature.ISS(X, SimpleWord("[11][122]"))

    Args:
        word_string (str, optional): String representation of the word.
            Names of available letters can be used like
            ``[ABS(1)SIMPLE(2)][ABS(1)]`` to create the corresponding
            word.
    """

    def __init__(self, word_string: Optional[str] = None) -> None:
        self._alpha: Union[float, list[float]] = 0.0
        self._extended_letters: list[ExtendedLetter] = []
        if word_string is not None:
            self.multiply(word_string)

    @property
    def alpha(self) -> list[float]:
        """Penalization term for the calculation of iterated sums.
        Sums that use multiplication of indices that are further away
        are scaled down. The ammount of downscaling depends on this
        alpha.
        Its default value is 0, meaning no scaling. The value can be
        changed for each contiguous extended letter pair in the word
        respectively (given as a list of length ``len(word)-1``) or by
        choosing the same value once for the whole list.
        """
        if isinstance(self._alpha, float):
            return [self._alpha] * (len(self) - 1)
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Union[float, list[float]]) -> None:
        if isinstance(alpha, list):
            if len(alpha) != len(self) - 1:
                raise ValueError("alpha has to have the same length as " +
                                 "number of extended letters in the word - 1")
        elif not isinstance(alpha, float):
            raise ValueError("alpha has to be a float or list of floats")
        self._alpha = alpha

    def multiply(self, other: Union["Word", ExtendedLetter, str]) -> None:
        """Appends one or more extended letters to the word. A group of
        extended letters have to be given as another Word object or as
        a string.
        """
        if isinstance(other, ExtendedLetter):
            self._extended_letters.append(other)
        elif isinstance(other, Word):
            for el in other._extended_letters:
                self._extended_letters.append(el)
        elif isinstance(other, str):
            els_raw = other.split("]")[:-1]
            for el_raw in els_raw:
                self.multiply(ExtendedLetter(el_raw[1:]))
        else:
            raise TypeError(f"Cannot multiply Word with {type(other)}")

    def copy(self) -> "Word":
        sw = Word()
        sw._extended_letters = [el.copy() for el in self._extended_letters]
        return sw

    def __len__(self) -> int:
        return len(self._extended_letters)

    def __iter__(self) -> "Word":
        self._el_iterator_index = -1
        return self

    def __next__(self):
        if self._el_iterator_index < len(self._extended_letters)-1:
            self._el_iterator_index += 1
            return self._extended_letters[self._el_iterator_index]
        raise StopIteration()

    def __copy__(self) -> "Word":
        return self.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            raise NotImplementedError
        return False

    def __str__(self) -> str:
        return "".join([str(el) for el in self._extended_letters])

    def __repr__(self) -> str:
        return f"fruits.words.word.Word('{self!s}')"


class SimpleWord(Word):
    """A Word that is directly inheriting from
    :class:`~fruits.words.word.Word`.

    A simple word is a special form of an ``Word`` that contains
    letters (i.e. functions) that extract a single dimension of a
    multidimesional time series.
    The letters in this object do the same as the predefined letter
    :meth:`~fruits.signature.letters.simple_letter` but they are saved as
    integers in a numpy array which allows for better space and
    computation management. It is therefore used to speed up the
    calculation of iterated sums and to make the creation of such a
    construct easier.

    Example:

    .. code-block:: python

        X = numpy.random.randint(-100, 100, (2, 100))
        word = SimpleWord("[11][122]")
        iterated_sums = fruits.signature.ISS(X, word)

    This code does the same as

    .. code-block:: python

        inner = numpy.cumsum(X[0, :]**2)
        inner = np.roll(inner, 1)
        inner[0] = 0
        iterated_sums = numpy.cumsum(inner * X[0, :]*X[1, :]**2)

    Each bracket pair is interpreted as an extended letter. Each
    extended letter is equal to all of its letter-permutations.
    This means that::

        SimpleWord("[12][122]") == SimpleWord("[21][212]")

    is true.

    The extended letters are saved internally as lists with a length
    equal to the highest number in the extended letter. The word
    ``"[12][122]"`` will be transformed to ``[[1,1], [1,2]]``. An entry
    ``e_i`` in one of the inner lists ``L_j`` corresponds to the number
    of occurences of letter ``i`` in the ``j``-th extended letter.

    Enclose dimensions with normal brackets that have two or more
    digits, like ``SimpleWord("[122(10)(62)][(24)5]")``.

    Args:
        string (str): Will be used to create the SimpleWord as
            shown in the example above. It has to match the regular
            expression ``(\\[(\\d|\\(\\d+\\))+\\])+``.
    """

    def __init__(self, string: str):
        super().__init__()
        self._extended_letters: list[list[int]] = []
        self._max_dim = 0
        self._name = ""
        self.multiply(string)

    def multiply(self, other: Union[Word, ExtendedLetter, str]):
        """Multiplies another word with the SimpleWord object.
        The word is given as a string matching the examples given in
        the class definition.
        """
        if not isinstance(other, str):
            raise NotImplementedError
        if not re.fullmatch(r"(\[(\d|\(\d+\))+\])+", other):
            raise ValueError("SimpleWord can only be multiplied with a "
                             "string matching the regular expression "
                             r"'(\[(\d|\(\d+\))+\])+'")
        self._name = self._name + other
        els_raw = [x[1:] for x in other.split("]")][:-1]
        els_int: list[list[int]] = []
        for i in range(len(els_raw)):
            els_int.append([])
            j = 0
            while j < len(els_raw[i]):
                if els_raw[i][j] == "(":
                    temp = ""
                    j += 1
                    while els_raw[i][j] != ")":
                        temp += els_raw[i][j]
                        j += 1
                    if temp == "":
                        temp = "1"
                    els_int[-1].append(int(temp))
                else:
                    els_int[-1].append(int(els_raw[i][j]))
                j += 1
        max_dim = max([letter for el_int in els_int for letter in el_int])
        if max_dim > self._max_dim:
            for el in self._extended_letters:
                for i in range(max_dim-self._max_dim):
                    el.append(0)
            self._max_dim = max_dim
        for el_int in els_int:
            el = [0 for i in range(max_dim)]
            for letter in set(el_int):
                el[letter-1] = el_int.count(letter)
            self._extended_letters.append(el)

    def copy(self) -> "SimpleWord":
        sw = SimpleWord(self._name)
        sw._extended_letters = [el.copy() for el in self._extended_letters]
        return sw

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleWord):
            raise NotImplementedError
        return list(self._extended_letters) == list(other._extended_letters)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"fruits.words.word.SimpleWord('{self!s}')"
