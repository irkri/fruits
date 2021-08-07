import re
from abc import ABC, abstractmethod

from fruits.core.letters import ExtendedLetter

class AbstractWord(ABC):
    """This abstractly used class is a word, that is collection of
    extended letters. An extended letter is a collection of letters.
    A letter is a function that accepts and returns numpy arrays. The
    order of the letters in an extended letter doesn't matter.

    An AbstractWord object is used for the calculation of iterated sums
    for a list of numbers. This can be done by calling::

        fruits.core.iss.ISS(X, word)

    where ``word`` is an object of a class that inherits AbstractWord.

    Each AbstractWord is iterable. The items returned by the iterator
    are the extended letters (``fruits.core.letters.ExtendedLetter``).

    One can extend an already created word by calling
    ``self.multiply(...)``.

    :param name: Name of the AbstractWord object, defaults to ""
    :type name: str, optional
    """
    def __init__(self, name: str = ""):
        self.name = name

    @property
    def name(self) -> str:
        """Simple identifier string for the object.

        This property also has a setter method.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractmethod
    def multiply(self, other):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def __copy__(self):
        return self.copy()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "fruits.core.wording.AbstractWord"


class ComplexWord(AbstractWord):
    """A ComplexWord is directly inherited from ``AbstractWord``.
    
    This class manages the concatenation of ``ExtendedLetter`` objects.

    Example:

    .. code-block:: python

        word = ComplexWord()

        el01 = ExtendedLetter()
        el01.append(fruits.core.letters.simple_letter, 0)
        el01.append(fruits.core.letters.simple_letter, 0)
        el02 = ExtendedLetter()
        el02.append(fruits.core.letters.simple_letter, 0)
        el02.append(fruits.core.letters.simple_letter, 1)
        el02.append(fruits.core.letters.simple_letter, 1)

        word.multiply(el01)
        word.multiply(el02)

        iterated_sums = fruits.core.ISS(X, word)

    The result in ``iterated_sums`` is equal to 

    .. code-block:: python

        numpy.cumsum(numpy.cumsum(X[0, :]**2) * X[0, :]*X[1, :]**2)

    which corresponds to a call of
    ``fruits.core.ISS(X, SimpleWord("[11][122]"))``.

    :param name: Name of the object, has no influence on any
        computation., defaults to ""
    :type name: str, optional
    """
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._extended_letters = []

    def multiply(self, other):
        """Appends ExtendedLetter objects to this word.
        
        :type other: ComplexWord or ExtendedLetter
        """
        if isinstance(other, ExtendedLetter):
            self._extended_letters.append(other)
        elif isinstance(other, ComplexWord):
            for el in other._extended_letters:
                self._extended_letters.append(el)
        else:
            raise TypeError(f"Cannot multiply ComplexWord with {type(other)}")

    def copy(self):
        """Returns a copy of this word.
        
        :rtype: ComplexWord
        """
        sw = ComplexWord(self.name)
        sw._extended_letters = [el.copy() for el in self._extended_letters]
        return sw

    def __iter__(self):
        self._el_iterator_index = -1
        return self

    def __next__(self):
        if self._el_iterator_index < len(self._extended_letters)-1:
            self._el_iterator_index += 1
            return self._extended_letters[self._el_iterator_index]
        raise StopIteration()

    def __eq__(self, other):
        return False

    def __str__(self) -> str:
        return "".join([str(el) for el in self._extended_letters])

    def __repr__(self) -> str:
        return "fruits.core.wording.ComplexWord"


class SimpleWord(AbstractWord):
    """A SimpleWord is directly inherited from ``AbstractWord``.

    A ``SimpleWord`` is a special form of an ``AbstractWord`` that
    contains functions which extract a single dimension of a
    multidimesional time series.
    The letters in this object do the same like the predefined letter
    ``fruits.core.letters.simple_letter`` but they are saved as integers
    in a numpy array which allows for better space and computation
    management. It is therefore used to speed up the calculation of
    iterated sums and to make the creation of such a construct easier.
    Defining simple words should always be achieved by creating a
    ``SimpleWord`` object rather than defining a ``ComplexWord`` and
    using the mentioned ``simple_letter``.

    Example:

    .. code-block:: python

        X = numpy.random.randint(-100, 100, (2, 100))
        word = SimpleWord("[11][122]")
        iterated_sums = fruits.core.ISS(X, word)

    The result ``iterated_sums`` is now equal to 

    .. code-block:: python

        numpy.cumsum(numpy.cumsum(X[0, :]**2) * X[0, :]*X[1, :]**2).

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

    :param string: Will be used to create the SimpleWord as
        shown in the example above. It has to match the regular
        expression ``([d+])+`` where ``d+`` denotes one or more digits.
        This string will also be used to set the name of the object.
    :type string: str
    """
    def __init__(self, string: str):
        super().__init__()
        self._extended_letters = []
        self._max_dim = 0
        self.multiply(string)

    def multiply(self, string: str):
        """Multiplies another word with the SimpleWord object.
        The word is given as a string matching the regular expression
        ``([d+])+`` where ``d+`` denotes one or more digits.
        
        :type string: str
        """
        if (not isinstance(string, str) or
            not re.fullmatch(r"(\[\d+\])+", string)):
            raise ValueError("SimpleWord can only be multiplied with a "+
                             "string matching the regular expression "+
                             r"'(\[\d+\])+'")
        self.name = self.name + string
        els_raw = [x[1:] for x in string.split("]")][:-1]
        max_dim = max([int(letter) for el_raw in els_raw for letter in el_raw])
        if max_dim > self._max_dim:
            for el in self._extended_letters:
                for i in range(max_dim-self._max_dim):
                    el.append(0)
            self._max_dim = max_dim
        for el_raw in els_raw:
            el = [0 for i in range(max_dim)]
            for letter in set(l for l in el_raw):
                el[int(letter)-1] = el_raw.count(letter)
            self._extended_letters.append(el)

    def copy(self):
        """Returns a copy of this SimpleWord.
        
        :rtype: SimpleWord
        """
        sw = SimpleWord(self.name)
        sw._extended_letters = [el.copy() for el in self._extended_letters]
        return sw

    def __iter__(self):
        self._el_iterator_index = -1
        return self

    def __next__(self):
        if self._el_iterator_index < len(self._extended_letters)-1:
            self._el_iterator_index += 1
            return self._extended_letters[self._el_iterator_index]
        raise StopIteration()

    def __eq__(self, other):
        if not isinstance(other, SimpleWord):
            raise TypeError(f"Cannot compare SimpleWord with {type(other)}")
        return list(self._extended_letters) == list(other._extended_letters)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "fruits.core.wording.SimpleWord"
