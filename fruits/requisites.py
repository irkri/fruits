import numpy as np

from fruits.preparation.abstract import DataPreparateur
from fruits.preparation.transform import INC
from fruits.core.iss import ISS
from fruits.core.wording import AbstractWord, SimpleWord

class Requisite:
    """A class used to calculate requisites for objects that can be
    added to a :class:`~fruits.base.fruit.Fruit`. A requisite then takes
    a time series dataset and calculates up to the first two steps of a
    FRUITS pipeline.
    It can be configured by specifying which
    :class:`~fruits.preparation.abstract.DataPreparateur` and
    :class:`~fruits.preparation.wording.AbstractWord` to use.

    .. code-block::python
        req = Requisite("monotone")
        req.configure(preparateur=fruits.preparation.INC(),
                      word=fruits.core.SimpleWord("[11]"))
        req.process(X)
        result = req.get()

    The above steps calculate the increments of each time series in the
    dataset ``X`` and return the iterated sums for the word "[11]".
    Two Requisite objects are equal if their preparateur and word are
    equal. This equality can be checked by using the operator ``==``.

    :param ident: Identification of the object. This value has to be
        unique under all used Requisite objects. Have a look at all
        defined Requisite objects in
        :func:`~fruits.requisites.get_available`.
    :type ident: str
    """
    def __init__(self, ident: str):
        self._preparateur = None
        self._word = None
        self._ident = ident

    def configure(self,
                  preparateur: DataPreparateur = None,
                  word: AbstractWord = None):
        """Specifies which preparateur and word to use for future
        processing.
        
        :type preparateur: DataPreparateur, optional
        :type word: AbstractWord, optional
        """
        self._preparateur = preparateur
        self._word = word

    def process(self, X: np.ndarray):
        """Processes the given time series dataset.

        :type X: np.ndarray
        """
        result = X.copy()
        if self._preparateur is not None:
            result = self._preparateur.fit_prepare(result)
        if self._word is not None:
            result = ISS(result, self._word)
        return result

    def is_empty(self) -> bool:
        """Returns ``True`` if no preparateur and word are specified,
        else ``False``.
        
        :rtype: bool
        """
        if self._preparateur is None and self._word is None:
            return True
        return False

    def __eq__(self, other) -> bool:
        if not isinstance(other, Requisite):
            raise TypeError(f"Cannot compare Requisite with {type(other)}")
        if (self._preparateur == other._preparateur and
            self._word == other._word):
            return True
        return False

    def __str__(self) -> str:
        return self._ident


# definition of standard requisites

_Monotone = Requisite("INC -> [11]")
_Monotone.configure(preparateur=INC(zero_padding=False),
                    word=SimpleWord("[11]"))

_AVAILABLE = {
    str(_Monotone): _Monotone,
}

def get_available() -> list:
    """Returns a list of all available requisites. Add new ones by
    using :meth:`~fruits.requisites.log`.
    
    :rtype: list
    """
    return _AVAILABLE.items()

def get(requisite_ident: str) -> Requisite:
    """Returns the Requisite object for the given identification string.
    
    :type requisite_ident: str
    :rtype: Requisite
    """
    return _AVAILABLE[requisite_ident]

def log(requisite: Requisite):
    """Logs a given requisite as 'available' for later usage.

    :type requisite: Requisite
    """
    if str(requisite) in _AVAILABLE:
        raise ValueError("Requisite with matching identification "+
                         "string found")
    _AVAILABLE[str(requisite)] = requisite


class RequisiteContainer:
    """Class that is used in a :class:`~fruits.base.fruit.Fruit`.
    It processes every requisite the configuration needs.
    """
    def __init__(self):
        self._requisites = {}

    def register(self, requisite_ident: str):
        """Registers a requisite by its string to the container.
        If the requisite already is registered, do nothing.
        Get possible requisites by calling
        :func:`~fruits.requisites.get_available`.
        
        :type req_identification: str
        """
        if not requisite_ident in self._requisites:
            if not requisite_ident in _AVAILABLE:
                raise ValueError("No requisite with the given "+
                                   "identification found")
            self._requisites[requisite_ident] = None

    def process(self, X: np.ndarray):
        """Process time series dataset on every registered requisite.
        
        :type X: np.ndarray
        """
        for req in self._requisites:
            self._requisites[req] = get(req).process(X)

    def get(self, requisite_ident: str) -> np.ndarray:
        """Returns the processed results of the requisite with the
        given identification.
        
        :type requisite_ident: str
        """
        if not requisite_ident in self._requisites:
            raise ValueError("Requisite was not registered")
        if (result := self._requisites[requisite_ident]) is not None:
            return result
        else:
            raise RuntimeError("Requisite is not processed")

    def clear(self):
        """Clears all cached results of registered requisites."""
        for req in self._requisites:
            self._requisites[req] = None
