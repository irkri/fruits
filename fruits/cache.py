import numpy as np

from fruits.core.iss import ISS

class FruitString:
    """A FruitString is used in a ``fruits.Fruit`` object to calculate
    prerequisites of feature sieves. This class is only used internally 
    but can also be used to connect a
    ``fruits.preparation.DataPreparateur`` with a
    ``fruits.core.AbstractWord`` to simulate the first two steps of a
    ``fruits.Fruit`` pipeline.

    .. code-block::python
        fruitstring = FruitString()
        fruitstring.preparateur = fruits.preparation.INC()
        fruitstring.word = fruits.core.SimpleWord("[11]")
        fruitstring.process(X)
        X_iss = fruitstring.get()

    The above steps calculate the increments of each time series in the
    dataset ``X`` and return the iterated sums ``X_iss`` of the word
    "[11]".
    Two FruitString objects are equal if their preparateur and word are
    equal. This equality can be checked by using the operator ``==``.
    """
    def __init__(self):
        self.preparateur = None
        self.word = None

        self._result = None

    def process(self, X: np.ndarray):
        """Calculated the results of
        ``fruits.core.ISS(self.preparateur.fit_prepare(X), self.word)``.
        
        :type X: np.ndarray
        """
        self._result = X
        if self.preparateur is not None:
            self._result = self.preparateur.fit_prepare(self._result)
        if self.word is not None:
            self._result = ISS(self._result, self.word)[:, 0, :]

    def get(self) -> np.ndarray:
        """Returns the saved results calculated with ``self.process``.
        
        :rtype: np.ndarray
        :raises: RuntimeError if ``self.process`` wasn't called
        """
        if self._result is None:
            raise RuntimeError("FruitString not processed")
        return self._result

    def is_empty(self) -> bool:
        """Returns ``True`` if preparateur and word are not specified
        yet, else ``False``.
        
        :rtype: bool
        """
        if self.preparateur is None and self.word is None:
            return True
        return False

    def is_processed(self) -> bool:
        """Returns ``True`` if ``self.process`` was called, else
        ``False``.
        
        :rtype: bool
        """
        if self._result is not None:
            return True
        return False

    def __eq__(self, other) -> bool:
        if not isinstance(other, FruitString):
            raise TypeError("Can only compare FruitString with another " +
                            "FruitString")
        if (self.preparateur == other.preparateur and
            self.word == other.word):
            return True
        return False
