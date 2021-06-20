import numpy as np

from fruits.core.iss import ISS

class FruitString:
    def __init__(self):
        self.preparateur = None
        self.word = None

        self._result = None

    def get(self, X: np.ndarray = None) -> np.ndarray:
        if self._result is not None:
            return self._result
        elif X is None:
            raise RuntimeError("No results available, X has to be supplied")

        self._result = X
        if self.preparateur is not None:
            self._result = self.preparateur.fit_prepare(self._result)
        if self.word is not None:
            self._result = ISS(self._result, self.word)[:, 0, :]
        return self._result

    def is_empty(self) -> bool:
        if self.preparateur is None and self.word is None:
            return True
        else:
            return False

    def __eq__(self, other) -> bool:
        if not isinstance(other, FruitString):
            raise TypeError("Can only compare FruitString with another " +
                            "FruitString")
        if (self.preparateur == other.preparateur and
            self.word == other.word):
            return True
        return False
