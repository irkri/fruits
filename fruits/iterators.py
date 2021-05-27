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
    ```
    >>> iterator = SummationIterator()
    >>> iterator.append_monomials([lambda X: X[0, :]**2,
                                   lambda X: X[1, :]**3],
                                  [lambda X: X[0, :]])
    >>> fruits.core.ISS(X, [iterator])
    ```
    The result of this last function call is
    `CS(CS(X[0, :]**2 * X[1, :]**3)) * X[0, :])`,
    where CS denotes the function that calulcates the cumulative sums.
    
    :param name: Name for the object, defaults to ""
    :type name: str, optional
    :param scale: Scale of the object, uncommon option, defaults to 0
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

    def __repr__(self) -> str:
        return "SummationIterator('" + self.name + "')"

    def append_monomials(self, *objects):
        """Append a list of functions to the class. One list is
        interpreted as a monomial in the SummationIterator.
        
        :param *objects: List(s) of functions
        :type *objects: list or multiple lists
        :raises: ValueError if monomials aren't lists
        """
        for obj in objects:
            if not instances(obj, list):
                raise ValueError("Monomials have to be given in form of lists")
            self._monomials.append(obj)

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


class SimpleWord(SummationIterator):
    """Class SimpleWord

    A SimpleWord is a special form of a SummationIterator that contains
    functions which extract a single dimension of a multidimesional 
    time series.
    It is used to speed up the calculation of iterated sums.

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
            m = np.zeros(max_dim + 1, dtype=np.int32)
            for i in range(max_dim+1):
                m[i] = monomial.count(i)
            self.append_monomials(m)

    def monomials(self):
        """Returns a generator of all monomials in this object.
        
        :returns: Generator of monomials
        :rtype: generator of lists
        """
        for mon in self._monomials:
            yield mon

    def __repr__(self) -> str:
        return "SimpleWord(" + self.name + ")"

    def copy(self):
        """Returns a copy of this SimpleWord.
        
        :returns: Copy of this object
        :rtype: SimpleWord
        """
        si = SimpleWord(self.name)
        si._monomials = [x.copy() for x in self._monomials]
        si.scale = self.scale
        return si

    def __copy__(self):
        return self.copy()


def generate_random_words(number: int,
                          dim: int = 1,
                          monomial_length: int = 3,
                          n_monomials: int = 3) -> list:
    """Returns randomly initialized instances of the class SimpleWord.
    
    :param number: number of instances created
    :type number: int
    :param dim: maximal dimensionality the letters of any Monomial in 
    any SimpleWord can extract, defaults to 1
    :type dim: int, optional
    :param monomial_length: maximal number of letters of any Monomial, 
    defaults to 3
    :type monomial_length: int, optional
    :param n_monomials: maximal number of Monomials of any SimpleWord, 
    defaults to 3
    :type n_monomials: int, optional
    :returns: List of SimpleWords
    :rtype: list
    """
    words = []
    av_elements = [str(i+1) for i in range(dim)]
    for i in range(number):
        length = np.random.randint(1, n_monomials+1)
        conc = ""
        for j in range(length):
            clength = np.random.randint(1, monomial_length+1)
            conc += "[" + "".join(np.random.choice(av_elements, size=clength))
            conc += "]"
        words.append(SimpleWord(conc))
    return words

def generate_words(dim: int = 1,
                   monomial_length: int = 1,
                   n_monomials: int = 1) -> list:
    """Returns all possible and unique SimpleWords up to the given 
    boundaries.
    
    :param dim: maximal dimensionality the letters of any Monomial in 
    any SimpleWord can extract, defaults to 1
    :type dim: int, optional
    :param monomial_length: maximal number of letters of any Monomial, 
    defaults to 1
    :type monomial_length: int, optional
    :param n_monomials: maximal number of Monomials of any SimpleWord, 
    defaults to 1
    :type n_monomials: int, optional
    :returns: List of SimpleWords
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
