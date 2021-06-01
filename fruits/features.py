from abc import ABC, abstractmethod

import numpy as np

from fruits import accelerated, core

class FeatureSieve(ABC):
    """Abstract class FeatureSieve
    
    A FeatureSieve object is used to extract a single number out of an
    multidimensional numpy array.
    Each class that inherits FeatureSieve must override the methods
    `FeatureSieve.fit` and `FeatureSieve.sieve`.
    """
    def __init__(self, name: str = ""):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Simple identifier for a FeatureSieve object without any
        computational meaninng.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        out = "FeatureSieve('" + self._name + "')"
        return out

    @abstractmethod
    def copy(self):
        """Returns a copy of this FeatureSieve object.

        :returns: Copy of this object
        :rtype: FeatureSieve
        """
        pass

    def __copy__(self):
        return self.copy()

    def fit(self, X: np.ndarray):
        """Fits the FeatureSieve to the dataset. This method may do
        nothing for some classes that inherit FeatureSieve.

        :param X: (multidimensional) time series dataset
        :type X: np.ndarray
        """
        pass

    @abstractmethod
    def sieve(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_sieve(self, X: np.ndarray) -> np.ndarray:
        """Equivalent of calling `FeatureSieve.fit` and
        `FeatureSieve.sieve` consecutively.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: array of features (one for each time series)
        :rtype: np.ndarray
        """
        self.fit(X)
        return self.sieve(X)


class PPV(FeatureSieve):
    """FeatureSieve: Proportion of positive values
    
    For a calculated quantile with `PPV.fit`, this FeatureSieve
    calculates the proportion of values in a time series that is greater
    than the calculated quantile.

    :param quantile: Quantile as actual value or probability for
        quantile calculation (e.g. 0.5 for the 0.5-quantile),
        defaults to 0.5
    :type quantile: float, optional
    :param constant: if `True`, the argument `quantile` is interpreted
        as the actual value for the quantile, defaults to False
    :type constant: bool, optional
    :param sample_size: Sample size to use for quantile calculation.
        This option can be ignored if `constant` is set to `True`,
        defaults to 0.05
    :type sample_size: float, optional
    :param name: Name for the object,
        defaults to "Proportion of positive values"
    :type name: str, optional
    """
    def __init__(self,
                 quantile: float = 0.5,
                 constant: bool = False,
                 sample_size: float = 0.05,
                 name: str = "Proportion of positive values"):
        super().__init__(name)
        self._q_input = quantile
        self._q = None
        self._constant = constant
        self._sample_size = sample_size
        if not constant and not (0 < self._q_input < 1):
            raise ValueError("If 'constant' is set to False, quantile "+
                             "has to be a value between 0 and 1")
        if not 0 < sample_size <= 1:
            raise ValueError("'sample_size' has to be a float between 0 and 1")

    def fit(self, X: np.ndarray):
        """Calculates and remembers the quantile of the time series
        data.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        """
        self._q = self._q_input
        if not self._constant:
            sample_size = int(self._sample_size * len(X))
            if sample_size < 1:
                sample_size = 1
            selection = np.random.choice(np.arange(len(X)),
                                         size=sample_size,
                                         replace=False)
            self._q = np.quantile(np.array(
                                    [X[i] for i in selection]
                                  ).flatten(),
                                  self._q_input)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns `sum(X>=q)/len(X)` if q denotes the quantile
        calculated with `self.fit`.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: array of features (one for each time series)
        :rtype: np.ndarray
        :raises: RuntimeError if `self.fit` wasn't called
        """
        if self._q is None:
            raise RuntimeError("Missing call of PPV.fit()")
        return accelerated._fast_ppv(X, self._q)

    def copy(self):
        """Returns a copy of this object.
        
        :returns: Copy of this object
        :rtype: PPV
        """
        fs = PPV(self._q_input,
                 self._constant,
                 self._sample_size,
                 self.name)
        return fs


class PPVC(PPV):
    """FeatureSieve: Proportion of connected components of positive
    values
    
    For a calculated quantile with `PPV.fit`, this FeatureSieve
    calculates the connected components of the proportion of values in
    a time series that is greater than the calculated quantile.
    This is equivalent to the number of consecutive strips of 1's in
    the array (X>=quantile).

    :param quantile: Quantile as actual value or probability for
        quantile calculation (e.g. 0.5 for the 0.5-quantile),
        defaults to 0.5
    :type quantile: float, optional
    :param constant: if `True`, the argument `quantile` is interpreted
        as the actual value for the quantile, defaults to False
    :type constant: bool, optional
    :param sample_size: Sample size to use for quantile calculation.
        This option can be ignored if `constant` is set to `True`,
        defaults to 0.05
    :type sample_size: float, optional
    :param name: Name for the object, defaults to
        "Proportion of connected components of positive values"
    :type name: str, optional
    """
    def __init__(self,
                 quantile: float = 0.5,
                 constant: bool = False,
                 sample_size: float = 0.05,
                 name:str = "Proportion of connected components of "+
                            "positive values"):
        super().__init__(quantile,
                         constant,
                         sample_size,
                         name)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the number of consecutive strips of 1's in `(X>=q)`
        if q denotes the quantile calculated with `self.fit`.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: array of features (one for each time series)
        :rtype: np.ndarray
        :raises: RuntimeError if `self.fit` wasn't called
        """
        if self._q is None:
            raise RuntimeError("Missing call of PPV.fit()")
        positive = np.pad((X >= self._q).astype(np.int32), 
                          ((0,0), (1,0)), 'constant')
        diff = positive[:, 1:] - positive[:, :-1]
        s = np.sum(diff == 1, axis=-1)
        # At most X.shape[1]/2 connected components are possible.
        return 2*s / X.shape[1]


class MAX(FeatureSieve):
    """FeatureSieve: Maximal value
    
    This FeatureSieve returns the maximal value for each time series in
    a given dataset.

    :param cut: If cut is an index of the time series array, the time
        series will be cut at this point before calculating the maximum.
        If it is a real number in (0,1), the corresponding coquantile
        will be calculated first and the result will be treated as the
        cutting index., defaults to -1
    :type cut: int, optional
    :param name: Name of the object, defaults to "Maximal value"
    :type name: str, optional
    """
    def __init__(self,
                 cut: int = -1,
                 name: str = "Maximal value"):
        super().__init__(name)
        self._cut = cut

    def fit(self, X: np.ndarray):
        """Fits the MAX feature sieve to the given dataset by
        calculating a cutting point if neccessary.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        """
        if self._cut >= X.shape[1]:
            self._cut = X.shape[1] - 1
        elif 0 < self._cut < 1:
            self._cut = accelerated._coquantile(X, self._cut)
        elif self._cut < 0:
            self._cut = X.shape[1]
        elif self._cut == 0:
            self._cut = 1

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns np.array([max(X[i,:]) for i in range(len(X))]).

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (one feature for each time series)
        :rtype: np.ndarray
        """
        return accelerated._max(X[:, :self._cut])

    def copy(self):
        """Returns a copy of this object.
        
        :returns: Copy of this object
        :rtype: MAX
        """
        fs = MAX(self._cut, self.name)
        return fs


class MIN(FeatureSieve):
    """FeatureSieve: Minimal value
    
    This FeatureSieve returns the minimal value for each time series in
    a given dataset.
    
    :param cut: If cut is an index of the time series array, the time
        series will be cut at this point before calculating the minimum.
        If it is a real number in (0,1), the corresponding coquantile
        will be calculated first and the result will be treated as the
        cutting index., defaults to -1
    :type cut: int, optional
    :param name: Name of the object, defaults to "Minimal value"
    :type name: str, optional
    """
    def __init__(self,
                 cut: int = -1,
                 name: str = "Minimal value"):
        super().__init__(name)
        self._cut = cut

    def fit(self, X: np.ndarray):
        """Fits the MIN feature sieve to the given dataset by
        calculating a cutting point if neccessary.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        """
        if self._cut >= X.shape[1]:
            self._cut = X.shape[1] - 1
        elif 0 < self._cut < 1:
            self._cut = accelerated._coquantile(X, self._cut)
        elif self._cut < 0:
            self._cut = X.shape[1]
        elif self._cut == 0:
            self._cut = 1

    def sieve(self, X: np.ndarray):
        """Returns np.array([min(X[i,:]) for i in range(len(X))]).

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (one feature for each time series)
        :rtype: np.ndarray
        """
        return accelerated._min(X[:, :self._cut])

    def copy(self):
        """Returns a copy of this object.
        
        :returns: Copy of this object
        :rtype: MIN
        """
        fs = MIN(self._cut, self.name)
        return fs


class END(FeatureSieve):
    """FeatureSieve: Last value
    
    This FeatureSieve returns the last value of each time series in a
    given dataset.

    :param cut: If cut is an index of the time series array, this sieve
        will extract the value of the time series at this position. If
        it is a real number in (0,1), the corresponding coquantile will
        be calculated first and the result will be treated as the value
        index., defaults to -1
    :type cut: int, optional
    :param name: Name of the object, defaults to "Last value"
    :type name: str, optional

    """
    def __init__(self,
                 cut: int = -1,
                 name: str = "Last value"):
        super().__init__(name)
        self._cut = cut

    def fit(self, X: np.ndarray):
        """Fits the END feature sieve to the given dataset by
        calculating a cutting point if neccessary.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        """
        if self._cut >= X.shape[1]:
            self._cut = X.shape[1] - 1
        elif 0 < self._cut < 1:
            self._cut = accelerated._coquantile(X, self._cut)
        elif self._cut < 0:
            self._cut = -1
        elif self._cut == 0:
            self._cut = 1

    def sieve(self, X: np.ndarray):
        """Returns np.array([X[i, -1] for i in range(len(X))]).

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (one feature for each time series)
        :rtype: np.ndarray
        """
        return X[:, self._cut]

    def copy(self):
        """Returns a copy of this object.
        
        :returns: Copy of this object
        :rtype: END
        """
        fs = END(self._cut, self.name)
        return fs


def get_ppv(n: int = 1,
            a: float = 0,
            b: float = 1, 
            constant: bool = False,
            sample_size: float = 0.05) -> list:
    """Returns a list of PPV feature sieves.
    
    :param n: Number of sieves (quantiles will be evenly spaced numbers
        between `a` and `b`), defaults to 1
    :type n: int, optional
    :param a: Left interval border where the quantiles will be drawn
        from, defaults to 0
    :type a: float, optional
    :param b: Left interval border where the quantiles will be drawn
        from, defaults to 1
    :type b: float, optional
    :param constant: if `True`, the quantiles will be interpreted as
        actual numbers, not percentiles, defaults to False
    :type constant: bool, optional
    :param sample_size: The quantiles will be calculated on this 
        proportion of the input data, Only matters if `constant` is set
        to `False`, defaults to 0.05
    :type sample_size: float, optional
    """
    return [PPV(q, constant=constant, sample_size=sample_size)
            for q in np.linspace(a, b, num=n)]
