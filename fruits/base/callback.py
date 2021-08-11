from abc import ABC

import numpy as np

class AbstractCallback(ABC):
    """Another class inheriting ``AbstractCallback`` can overwrite one
    or more of the class methods.
    
    The callback can then be used in a call of
    :meth:`~fruits.Fruit.transform`.
    """
    def on_next_branch(self):
        """Called every time the current FruitBranch in a Fruit object
        is switched.
        """
        pass

    def on_preparateur(self, X: np.ndarray):
        """Called after the calculation of prepared data for each
        ``DataPreparateur``.
        """
        pass

    def on_preparation_end(self, X: np.ndarray):
        """Called once after the calculation of the prepared data with
        the last ``DataPreparateur``.
        """
        pass

    def on_iterated_sum(self, X: np.ndarray):
        """Called for every iterated sum calculated (for each word)."""
        pass

    def on_sieve(self, X: np.ndarray):
        """Called after calculation of the features for one iterated
        sum.
        """
        pass

    def on_sieving_end(self, X: np.ndarray):
        """Called after the calculation of the features for each
        iterated sum.
        """
        pass
