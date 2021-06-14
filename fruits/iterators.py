import re
import itertools

import numpy as np

class SummationIterator:
    """Class SummationIterator

    This (mostly abstractly) used class is a collection of monomials.
    A monomial is a list of functions. These functions should accept
    and return numpy arrays. The results from each monomial are
    multiplied before the cumulative sums are calculated.
    This happens in fruits.core.ISS.
    
    Example:

    .. code-block:: python

        iterator = SummationIterator()
        iterator.multiply([lambda X: X[0, :]**2, lambda X: X[1, :]**3])
        iterator.multiply([lambda X: X[0, :])
        fruits.core.ISS(X, [iterator])
    
    The result of this last function call is
    ``CS(CS(X[0, :]**2 * X[1, :]**3)) * X[0, :])``,
    where CS denotes the function that calulcates the cumulative sums.

    :param name: Name for the object, defaults to ""
    :type name: str, optional
    :param scale: Scale of the object, uncommon option. If an integer is
        given, the output the function fruits.core.ISS produces with
        this object as argument will be divided by 
        ``[time series length]/scale``., defaults to 0
    :type scale: int, optional
    """
    def __init__(self, name: str = "", scale: int = 0):
        self.name = name
        # list of lists (inner lists are monomials)
        self._monomials = []
        self.scale = scale

    @property
    def name(self) -> str:
        """Simple Identifier for the SummationIterator."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def scale(self) -> int:
        """Results from iterated sums may be to large and turn to inf or
        nan values very quick. Therefore we divide the results in each
        step of ISS(X, iterators) by [length of the time series]**scale.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: int):
        self._scale = scale

    def multiply(self, obj):
        """Appends a list of functions to the class. This list is
        interpreted as a monomial in the SummationIterator.
        
        :param monomial: List of functions
        :type monomial: list
        :raises: ValueError if monomials aren't lists
        """
        if isinstance(obj, list):
            self._monomials.append(obj)
        elif isinstance(obj, SummationIterator):
            for monomial in obj.monomials():
                self._monomials.append(monomial)
        else:
            raise ValueError("SummationIterator can only be mutliplied with "+
                             "a list or another SummationIterator")

    def monomials(self):
        """Returns a generator of all monomials in this object.
        
        :returns: Generator of monomials
        :rtype: generator of lists
        """
        for mon in self._monomials:
            yield mon

    def copy(self):
        """Returns a copy of this SummationIterator.
        
        :returns: Copy of this object
        :rtype: SummationIterator
        """
        si = SummationIterator(self.name)
        si._monomials = [x.copy() for x in self._monomials]
        si.scale = self.scale
        return si

    def __copy__(self):
        return self.copy()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "SummationIterator('" + self.name + "')"


class SimpleWord(SummationIterator):
    """Class SimpleWord

    A SimpleWord is a special form of a SummationIterator that contains
    functions which extract a single dimension of a multidimesional 
    time series.
    It is used to speed up the calculation of iterated sums.
    Once a SimpleWord is created, it is no longer possible to change
    its monomials or append new ones.

    It is possible to check if two SimpleWords are equal using the
    builtin '==' operator.

    :param string: Name for the object that is also used to create
        the monomials of the SummationIterator, e.g. '[11][221][1122]'
    :type string: str
    :param scale: Scale of the object, uncommon option, defaults to 0
    :type scale: int, optional
    :raises: ValueError if `string` doesn't match the regular
        expression '([d+])+' where 'd+' denotes one or more digits.
    """
    def __init__(self, string: str, scale: int = 0):
        super().__init__(string, scale)
        if not re.fullmatch(r"(\[\d+\])+", string):
            raise ValueError("SimpleWord can only be initilized with a "+
                             "string matching the regular expression "+
                             r"'(\[\d+\])+'")
        monomials_raw = [x[1:] for x in string.split("]")][:-1]
        max_dim = 0
        monomials = []
        for monomial_raw in monomials_raw:
            monomial = []
            for letter in monomial_raw:
                if int(letter)-1 > max_dim:
                    max_dim = int(letter) - 1
                monomial.append(int(letter)-1)
            monomials.append(monomial)
        self.max_dim = max_dim

        for monomial in monomials:
            m = []
            for i in range(max_dim+1):
                m.append(monomial.count(i))
            self.multiply(m)

    def monomials(self):
        """Returns a generator of all monomials in this object.
        
        :returns: Generator of monomials
        :rtype: generator of lists
        """
        for mon in self._monomials:
            yield mon

    def copy(self):
        """Returns a copy of this SimpleWord.
        
        :returns: Copy of this object
        :rtype: SimpleWord
        """
        si = SimpleWord(self.name)
        si._monomials = [x.copy() for x in self._monomials]
        si.scale = self.scale
        return si

    def __eq__(self, other):
        return list(self.monomials()) == list(other.monomials())

    def __copy__(self):
        return self.copy()

    def __repr__(self) -> str:
        return "SimpleWord(" + self.name + ")"


def generate_words(dim: int = 1,
                   monomial_length: int = 1,
                   n_monomials: int = 1) -> list:
    """Returns all possible and unique SimpleWords up to the given 
    boundaries.
    
    :param dim: Maximal dimensionality the letters of any monomial in 
        any SimpleWord can extract., defaults to 1
    :type dim: int, optional
    :param monomial_length: Maximal number of letters in any monomial., 
        defaults to 1
    :type monomial_length: int, optional
    :param n_monomials: Maximal number of monomials in any SimpleWord., 
        defaults to 1
    :type n_monomials: int, optional
    :returns: List of SimpleWord objects
    :rtype: list
    """
    monomials = []
    for l in range(1, monomial_length+1):
        mons = list(itertools.combinations_with_replacement(
                            list(range(1, dim+1)), l))
        for mon in mons:
            monomials.append(list(mon))

    words = []
    for n in range(1, n_monomials+1):
        words_n = list(itertools.product(monomials, repeat=n))
        for word in words_n:
            words.append("".join([str(x).replace(", ","") for x in word]))

    for i in range(len(words)):
        words[i] = SimpleWord(words[i])

    return words

def generate_words_of_length(l: int,
                             dim: int = 1) -> list:
    """Returns a list of all possible and unique SimpleWords that have
    exactly the given number of letters.
    For ``l=2`` and ``dim=2`` this will return a list containing::

        SimpleWord("[11]"), SimpleWord("[12]"), SimpleWord("[22]"),
        SimpleWord("[1][1]"), SimpleWord("[1][2]"), SimpleWord("[2][2]")
    
    :param l: Number of letters the words should contain
    :type l: int
    :param dim: Highest dimension of a letter., defaults to 1
    :type dim: int, optional
    """
    # generate all monomials that can occured in a word with exactly
    # l letters
    monomials = []
    for length in range(1, l+1):
        monomials.append([])
        mons = itertools.combinations_with_replacement(
                                list(range(1, dim+1)), length)
        for mon in mons:
            monomials[-1].append(list(mon))

    # generate all possible combinations of the monomials created above 
    # such that the combination is a word with l letters
    # (next line is in O(2^l), maybe find a better option later)
    choose_monomials = [t for i in range(1, l+1) for t in
                        itertools.combinations_with_replacement(
                            list(range(1, l+1)), i)
                        if sum(t)==l]

    # use the combinations above for generating all possible words with
    # l letters by using the monomials from the beginning
    words = []
    for choice in choose_monomials:
        selected_monomials = []
        for perm in set(itertools.permutations(choice)):
            selected_monomials.append(list(itertools.product(
                *[monomials[i-1] for i in perm])))
        for inner_words in [[str(list(x))[1:-1].replace(", ","")
                             for x in mon] for mon in selected_monomials]:
            for word in inner_words:
                words.append(word)

    for i in range(len(words)):
        words[i] = SimpleWord(words[i])

    return words
