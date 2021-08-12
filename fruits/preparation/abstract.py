from abc import abstractmethod

import numpy as np

from fruits.node import FruitNode

class DataPreparateur(FruitNode):
    """Abstract class for a data preparateur.
    
    A preparateur can be fitted on a three dimensional numpy array
    (preferably containing time series data). The output of
    ``self.prepare`` is a numpy array that matches the shape of the
    input array.
    A class derived from DataPreparateur can be added to a
    ``fruits.Fruit`` object for the preprocessing step.
    """
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    @property
    def name(self) -> str:
        """Simple identifier for this object."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def fit(self, X: np.ndarray):
        """Fits the DataPreparateur to the given dataset.
        
        :type X: np.ndarray
        """
        pass

    @abstractmethod
    def prepare(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_prepare(self, X: np.ndarray) -> np.ndarray:
        """Fits the given dataset to the DataPreparateur and returns
        the preparated results.
        
        :param X: A (multidimensional) time series dataset.
        :type X: np.ndarray
        """
        self.fit(X)
        return self.prepare(X)

    @abstractmethod
    def copy(self):
        pass

    def __copy__(self):
        return self.copy()

    def __eq__(self, other) -> bool:
        return False

    def __repr__(self) -> str:
        return "fruits.preparation.DataPreparateur"
