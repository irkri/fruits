import numpy as np

from abc import ABC

class AbstractCallback(ABC):
    """Abstract class AbstractCallback
    
    Another class inheriting ``AbstractCallback`` can overwrite one or
    more of the following methods:
    
    on_next_branch(self)
    ^^^^^^^^^^^^^^^^^^^^
    Called every time the current FruitBranch in a Fruit object is
    switched.
    
    on_preparateur(self, X: np.ndarray)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Called after the calculation of prepared data for each
    ``DataPreparateur``.
    
    on_preparation_end(self, X: np.ndarray)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Called once after the calculation of the prepared data with the
    last ``DataPreparateur``.
    
    on_iterated_sum(self, X: np.ndarray)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Called for every iterated sum calculated (for each word).
    
    on_sieve(self, X: np.ndarray)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Called after calculation of the features for one iterated sum.
    
    on_sieving_end(self, X: np.ndarray)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Called after the calculation of the features for each iterated sum.
    
    The callback can then be used in a call of ``Fruit.transform``.
    """
    def on_next_branch(self):
        pass

    def on_preparateur(self, X: np.ndarray):
        pass

    def on_preparation_end(self, X: np.ndarray):
        pass

    def on_iterated_sum(self, X: np.ndarray):
        pass

    def on_sieve(self, X: np.ndarray):
        pass

    def on_sieving_end(self, X: np.ndarray):
        pass
