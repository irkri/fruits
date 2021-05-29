from abc import ABC, abstractmethod

import numpy as np

class DataPreparateur:
    """Abstract class DataPreperateur
    
    A DataPreparateur object can be called on a three dimensional numpy 
    array. The output should be a numpy array that matches the shape of 
    the input array.
    """
    def __init__(self, name: str = ""):
        self.name = name

    @property
    def name(self) -> str:
        """Simple identifier for this object."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: DataPreparateur
        """
        dp = DataPreparateur(self.name)
        return dp

    @abstractmethod
    def _prepare(self, X: np.ndarray) -> np.ndarray:
        pass

    def __copy__(self):
        return self.copy()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the DataPreparateur object to the input time series
        dataset.

        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        :returns: preprocessed dataset
        :rtype: np.ndarray
        """
        return self._prepare(X)

    def __repr__(self) -> str:
        return "DataPreparateur('" + self._name + "')"


class INC(DataPreparateur):
    """DataPreparateur: Increments
    
    For a time series 
    `X = [x_1, x_2, ..., x_n]`
    this class produces the output
    `X_inc = [0, x_2-x_1, x_3-x_2, ..., x_n-x_{n-1}]`.
    If `zero_padding` is set to `False`, then the 0 above will be
    replaced by `x_1`.
    """
    def __init__(self,
                 zero_padding: bool = True,
                 name: str = "Increments"):
        super().__init__(name)
        self._zero_padding = zero_padding

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if self._zero_padding:
            out = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
            pad_widths = [(0,0) for dim in range(3)]
            pad_widths[2] = (1,0)
            out = np.pad(out, pad_width=pad_widths, mode="constant")
        else:
            out = np.zeros(X.shape)
            out[:, :, 1:] = np.delete((np.roll(X, -1, axis=2) - X), -1, axis=2)
            out[:, :, 0] = X[:, :, 0]
        return out

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: INC
        """
        dp = INC(self._zero_padding, self.name)
        return dp


class STD(DataPreparateur):
    """DataPreparateur: Standardization
    
    For a time series `X` this class produces the output
    `X_std = (X-mean(X))/std(X)`.
    """
    def __init__(self,
                 name: str = "Standardization"):
        super().__init__(name)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i, j, :] = X[i, j, :] - np.mean(X[i, j, :])
                out[i, j, :] /= np.std(X[i, j, :])
        return out

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: STD
        """
        dp = STD(self.name)
        return dp


class NRM(DataPreparateur):
    """DataPreparateur: Normalization
    
    For a time series `X` this class produces the output
    `X_nrm = (X-min(X))/(max(X)-min(X))`.
    """    
    def __init__(self,
                 name: str = "Normalization"):
        super().__init__(name)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                mini = np.min(X[i, j, :])
                maxi = np.max(X[i, j, :])
                out[i, j, :] = (X[i, j, :]-mini) / (maxi-mini)
        return out

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: NRM
        """
        dp = NRM(self.name)
        return dp
