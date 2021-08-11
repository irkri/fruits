from abc import ABC, abstractmethod

import numpy as np

from fruits.cache import FruitString

class FeatureSieve(ABC):
    """Abstract class for a feature sieve. Sieves are the last part of a
    FRUITS pipeline.
    
    A feature sieve is used to transforms a twodimensional numpy
    array containing iterated sums into a onedimensional numpy array of
    features. The length of the resulting array can be determined by
    calling ``FeatureSieve.nfeatures``.

    Each class that inherits FeatureSieve must override the methods
    ``FeatureSieve.sieve`` and ``FeatureSieve.nfeatures``.
    """
    def __init__(self, name: str = ""):
        super().__init__()
        self._name = name
        self._prereqs = None

    @property
    def name(self) -> str:
        """Simple identifier for a FeatureSieve object without any
        computational meaning.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractmethod
    def nfeatures(self) -> int:
        pass

    def fit(self, X: np.ndarray):
        """Fits the sieve to the dataset.
        
        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        """
        pass

    @abstractmethod
    def sieve(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_sieve(self, X: np.ndarray) -> np.ndarray:
        """Equivalent of calling ``FeatureSieve.fit`` and
        ``FeatureSieve.sieve`` consecutively.
        
        :param X: 2-dimensional numpy array of iterated sums.
        :type X: np.ndarray
        :returns: Array of features
        :rtype: np.ndarray
        """
        self.fit(X)
        return self.sieve(X)

    @abstractmethod
    def _prerequisites(self) -> FruitString:
        pass

    def _load_prerequisites(self, fs: FruitString):
        self._prereqs = fs

    @abstractmethod
    def copy(self):
        pass

    def summary(self) -> str:
        return "FeatureSieve('" + self._name + "')"

    def __copy__(self):
        return self.copy()

    def __repr__(self) -> str:
        return "FeatureSieve('" + self._name + "')"
