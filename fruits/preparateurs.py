from abc import ABC, abstractmethod

import numpy as np

class DataPreparateur(ABC):
    """Abstract class DataPreperateur
    
    A DataPreparateur object can be fitted on a three dimensional numpy 
    array. The output of DataPreparateur.prepare is a numpy array that
    matches the shape of the input array.
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

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: DataPreparateur
        """
        dp = DataPreparateur(self.name)
        return dp

    def fit(self, X: np.ndarray):
        """Fits the DataPreparateur to the given dataset.
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        pass

    @abstractmethod
    def prepare(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_prepare(self, X: np.ndarray) -> np.ndarray:
        """Fits the given dataset to the DataPreparateur and returns
        the preparated results.
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        self.fit(X)
        return self.prepare(X)

    def __copy__(self):
        return self.copy()

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

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the increments of all time series in X.
        This is the equivalent of the convolution of X and [-1,1].
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        :returns: stepwise slopes of each time series in X
        :rtype: np.ndarray
        """
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
        self._means = None
        self._stds = None

    def fit(self, X: np.ndarray):
        """Fits the dataset X to the DataPreparateur by calculating
        the mean and standard deviation of each time series in X and
        storing the calculations for later usage in self.prepare().
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        self._means = np.zeros(X.shape[:2])
        self._stds = np.zeros(X.shape[:2])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self._means[i, j] = np.mean(X[i, j, :])
                self._stds[i, j] = np.std(X[i, j, :])

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns (X_i-mu)/v where mu is the calculated mean and v is
        the standard deviation in self.fit().
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        :returns: (standardized) dataset
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if self._means is None or self._stds is None:
            raise RuntimeError("Missing call of STD.fit")
        out = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i, j, :] = X[i, j, :] - self._means[i, j]
                out[i, j, :] /= self._stds[i, j]
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
        self._maxs = None
        self._mins = None

    def fit(self, X: np.ndarray):
        """Fits the dataset X to the DataPreparateur by calculating
        the maximum and minimum of each time series in X and storing
        both values for later usage in self.prepare().
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        self._maxs= np.max(X, axis=2)
        self._mins= np.min(X, axis=2)

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns (X_i-min)/(max-min) where max/min were calculated in
        a call of self.fit().
        
        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        :returns: (normalized) dataset
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if self._maxs is None or self._mins is None:
            raise RuntimeError("Missing call of NRM.fit")
        out = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i, j, :] = X[i, j, :] - self._mins[i, j]
                out[i, j, :] /= self._maxs[i, j] - self._mins[i, j]
        return out

    def copy(self):
        """Returns a copy of the DataPreparateur object.
        
        :returns: Copy of this object
        :rtype: NRM
        """
        dp = NRM(self.name)
        return dp
