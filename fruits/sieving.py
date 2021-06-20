from abc import ABC, abstractmethod

import numpy as np

from fruits import _accelerated
from fruits.cache import FruitString
from fruits.preparation import INC
from fruits.core.wording import SimpleWord

class FeatureSieve(ABC):
    """Abstract class FeatureSieve
    
    A FeatureSieve object is used to transforms a twodimensional numpy
    array into a onedimensional numpy array.
    The length of the resulting array can be determined by calling
    ``FeatureSieve.nfeatures`? .

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

    @abstractmethod
    def _prerequisites(self) -> FruitString:
        pass

    def _load_prerequisites(self, fs: FruitString):
        self._prereqs = fs

    @abstractmethod
    def copy(self):
        pass

    def __copy__(self):
        return self.copy()

    def __repr__(self) -> str:
        out = "FeatureSieve('" + self._name + "')"
        return out


class PPV(FeatureSieve):
    """FeatureSieve: Proportion of positive values
    
    For a calculated quantile with `PPV.fit`, this FeatureSieve
    calculates the proportion of values in a time series that is greater
    than the calculated quantile.

    :param quantile: Quantile `q` or list of quantiles `[q_1, ..., q_n]`
        as actual value(s) or probability for quantile calculation
        (e.g. 0.5 for the 0.5-quantile)., defaults to 0.5
    :type quantile: float or list of floats, optional
    :param constant: If `True`, the argument `quantile` is interpreted
        as the actual value for the quantile. If `quantile` is a list,
        then `constant` can be a list of booleans `[b_1, ..., b_n]` 
        where `b_i` explains the behaviour/interpretation of `q_i` or a
        single boolean that explains what every single `q_i` is (value
        or probability)., defaults to False
    :type constant: bool or list of bools, optional
    :param sample_size: Sample size to use for quantile calculation.
        This option can be ignored if `constant` is set to `True`.,
        defaults to 0.05
    :type sample_size: float, optional
    :param segments: If `True`, then the proportion of values within
        each two quantiles will be calculated. If `quantile` is a list,
        then this list will be sorted and the corresponding features
        will be

        .. code-block::python
            np.array([np.sum(q_{i-1} <= X[k] < q_i)]) / len(X[k])])

        where `k` is the index of the time series and `i` ranges from
        1 to n.
        If set to `False`, then the features will be

        .. code-block::python
            np.array([np.sum(X[k] <= q_i)]) / len(X[k])])

        with the same index rules., defaults to False
    :type segments: bool, optional
    :param name: Name for the object.,
        defaults to "Proportion of positive values"
    :type name: str, optional
    """
    def __init__(self,
                 quantile: float = 0.5,
                 constant: bool = False,
                 sample_size: float = 0.05,
                 segments: bool = False,
                 name: str = "Proportion of positive values"):
        super().__init__(name)
        if isinstance(quantile, list):
            if not isinstance(constant, list):
                constant = [constant for i in range(len(quantile))]
            elif len(quantile) != len(constant):
                raise ValueError("If 'quantile' is a list, then 'constant'"+
                                 " also has to be a list of same length or"+
                                 " a single boolean.") 
            for q, c in zip(quantile, constant):
                if not c and not (0 <= q <= 1):
                    raise ValueError("If 'constant' is set to False,"+
                                     " 'quantile' has to be a value in [0,1]")
        else:
            quantile = [quantile]
            if isinstance(constant, list):
                if len(constant) > 1:
                    raise ValueError("'constant' has to be a single boolean"+
                                     " if 'quantile' is a single float")
            else:
                constant = [constant]
        if segments:
            self._q_c_input = list(zip(list(set(quantile)),constant))
            self._q_c_input = sorted(self._q_c_input, key=lambda x: x[0])
        else:
            self._q_c_input = list(zip(quantile,constant))
        self._q = None
        if not 0 < sample_size <= 1:
            raise ValueError("'sample_size' has to be a float in (0, 1]")
        self._sample_size = sample_size
        if segments and len(quantile) == 1:
            raise ValueError("If 'segments' is set to `True` then 'quantile'"+
                             " has to be a list of length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :returns: number of features per time series
        :rtype: int
        """
        if self._segments:
            return len(self._q_c_input) - 1
        else:
            return len(self._q_c_input)

    def fit(self, X: np.ndarray):
        """Calculates and remembers the quantile(s) of the input data.

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        """
        self._q = [x[0] for x in self._q_c_input]
        for i in range(len(self._q)):
            if not self._q_c_input[i][1]:
                sample_size = int(self._sample_size * len(X))
                if sample_size < 1:
                    sample_size = 1
                selection = np.random.choice(np.arange(len(X)),
                                             size=sample_size,
                                             replace=False)
                self._q[i] = np.quantile(np.array(
                                            [X[i] for i in selection]
                                         ).flatten(),
                                         self._q[i])

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.
        
        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: array of features
        :rtype: np.ndarray
        :raises: RuntimeError if `self.fit` wasn't called
        """
        if self._q is None:
            raise RuntimeError("Missing call of PPV.fit()")
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            if self._segments:
                for j in range(1, len(self._q)):
                    result[i, j-1] = np.sum(np.logical_and(
                                                self._q[j-1] <= X[i],
                                                X[i] < self._q[j]))
                    result[i, j-1] /= X.shape[1]
            else:
                for j in range(len(self._q)):
                    result[i, j] = np.sum((X[i] >= self._q[j]))
                    result[i, j] /= X.shape[1]
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def _prerequisites(self) -> FruitString:
        return FruitString()

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: PPV
        """
        fs = PPV([x[0] for x in self._q_c_input],
                 [x[1] for x in self._q_c_input],
                 self._sample_size,
                 self._segments,
                 self.name)
        return fs

    def __str__(self) -> str:
        string = "PPV(" + \
                f"quantile={[x[0] for x in self._q_c_input]}, " + \
                f"constant={[x[1] for x in self._q_c_input]}, " + \
                f"sample_size={self._sample_size}, " + \
                f"segments={self._segments})"
        return string


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
                         False,
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
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(len(self._q)):
            diff = _accelerated._increments(np.expand_dims(
                                            (X >= self._q[i]).astype(np.int32),
                                            axis=1))[:, 0, :]
            # At most X.shape[1]/2 connected components are possible.
            result[:, i] = 2*np.sum(diff == 1, axis=-1) / X.shape[1]
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def _prerequisites(self) -> FruitString:
        return FruitString()

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: PPV
        """
        fs = PPVC([x[0] for x in self._q_c_input],
                  [x[1] for x in self._q_c_input],
                  self._sample_size,
                  self.name)
        return fs

    def __str__(self) -> str:
        string = "PPVC(" + \
                f"quantile={[x[0] for x in self._q_c_input]}, " + \
                f"constant={[x[1] for x in self._q_c_input]}, " + \
                f"sample_size={self._sample_size}, " + \
                f"segments={self._segments})"
        return string


class MAX(FeatureSieve):
    """FeatureSieve: Maximal value
    
    This FeatureSieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    'cut'.

    :param cut: If cut is an index of the time series array, the time
        series will be cut at this point before calculating the maximum.
        If it is a real number in (0,1), the corresponding coquantile
        will be calculated first and the result will be treated as the
        cutting index.
        'cut' can also be a list of floats or integers which will be
        treated in the same way., defaults to -1
    :type cut: int or list of integers, optional
    :param segments: If set to `True`, then the cutting indices will be
        sorted and treated as interval borders and the maximum in each
        interval will be sieved. The left interval border is reduced by
        1 before slicing. This means that an input of `cut=[1,5,10]`
        results in two features `max(X[k, 0:5])` and `max(X[k, 4:10])`
        for every time series `X[k]`.
        If set to `False`, then the left interval border is always 0.
        This results in one more feature., defaults to `False`
    :type segments: bool, optional
    :param name: Name of the object, defaults to "Maximal value"
    :type name: str, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False,
                 name: str = "Maximal value"):
        super().__init__(name)
        self._cut = cut if isinstance(cut, list) else [cut]
        if segments and len(self._cut) == 1:
            raise ValueError("If 'segments' is set to False, then 'cut'"+
                             " has to be a list length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :returns: number of features per time series
        :rtype: int
        """
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (two dimensional if there are more than
            one features per time series)
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            prereq = self._prerequisites().get(np.expand_dims(X, axis=1))
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            new_cuts = []
            for j in range(len(self._cut)):
                cut = self._cut[j]
                if 0 < cut < 1:
                    cut = np.sum((prereq[i] <= (prereq[i, -1] * cut)))
                    if cut == 0:
                        cut = 1
                elif not isinstance(cut, int):
                    raise TypeError("Cut has to be a float in (0,1) or an " +
                                    "integer")
                elif cut < 0:
                    cut = X.shape[1]
                elif cut > X.shape[1]:
                    raise IndexError("Cutting index out of range")
                new_cuts.append(cut)
            if self._segments:
                new_cuts = sorted(list(new_cuts))
                for j in range(1, len(new_cuts)):
                    result[i, j-1] = np.max(X[i, new_cuts[j-1]-1:new_cuts[j]])
            else:
                for j in range(len(new_cuts)):
                    result[i, j] = np.max(X[i, :new_cuts[j]])
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def _prerequisites(self) -> FruitString:
        fs = FruitString()
        for c in self._cut:
            if 0 < c < 1:
                fs.preparateur = INC(zero_padding=False)
                fs.word = SimpleWord("[11]")
                break
        return fs

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: MAX
        """
        fs = MAX(self._cut, self._segments, self.name)
        return fs

    def __str__(self) -> str:
        string = "MAX(" + \
                f"cut={self._cut}, " + \
                f"segments={self._segments})"
        return string


class MIN(FeatureSieve):
    """FeatureSieve: Minimum value
    
    This FeatureSieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    'cut'.

    :param cut: If cut is an index of the time series array, the time
        series will be cut at this point before calculating the minimum.
        If it is a real number in (0,1), the corresponding coquantile
        will be calculated first and the result will be treated as the
        cutting index.
        'cut' can also be a list of floats or integers which will be
        treated in the same way., defaults to -1
    :type cut: int or list of integers, optional
    :param segments: If set to `True`, then the cutting indices will be
        sorted and treated as interval borders and the minimum in each
        interval will be sieved. The left interval border is reduced by
        1 before slicing. This means that an input of `cut=[1,5,10]`
        results in two features `min(X[k, 0:5])` and `min(X[k, 4:10])`
        for every time series `X[k]`.
        If set to `False`, then the left interval border is always 0.
        This results in one more feature., defaults to `False`
    :type segments: bool, optional
    :param name: Name of the object, defaults to "Minimum value"
    :type name: str, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False,
                 name: str = "Minimum value"):
        super().__init__(name)
        self._cut = cut if isinstance(cut, list) else [cut]
        if segments and len(self._cut) == 1:
            raise ValueError("If 'segments' is set to False, then 'cut'"+
                             " has to be a list length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :returns: number of features per time series
        :rtype: int
        """
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (two dimensional if there are more than
            one features per time series)
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            prereq = self._prerequisites().get(np.expand_dims(X, axis=1))
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            new_cuts = []
            for j in range(len(self._cut)):
                cut = self._cut[j]
                if 0 < cut < 1:
                    cut = np.sum((prereq[i] <= (prereq[i, -1] * cut)))
                    if cut == 0:
                        cut = 1
                elif not isinstance(cut, int):
                    raise TypeError("Cut has to be a float in (0,1) or an " +
                                    "integer")
                elif cut < 0:
                    cut = X.shape[1]
                elif cut > X.shape[1]:
                    raise IndexError("Cutting index out of range")
                new_cuts.append(cut)
            if self._segments:
                new_cuts = sorted(list(new_cuts))
                for j in range(1, len(new_cuts)):
                    result[i, j-1] = np.min(X[i, new_cuts[j-1]-1:new_cuts[j]])
            else:
                for j in range(len(new_cuts)):
                    result[i, j] = np.min(X[i, :new_cuts[j]])
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def _prerequisites(self) -> FruitString:
        fs = FruitString()
        for c in self._cut:
            if 0 < c <  1:
                fs.preparateur = INC(zero_padding=False)
                fs.word = SimpleWord("[11]")
                break
        return fs

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: MIN
        """
        fs = MIN(self._cut, self._segments, self.name)
        return fs

    def __str__(self) -> str:
        string = "MIN(" + \
                f"cut={self._cut}, " + \
                f"segments={self._segments})"
        return string


class END(FeatureSieve):
    """FeatureSieve: Last value
    
    This FeatureSieve returns the last value of each time series in a
    given dataset.

    :param cut: If cut is an index of the time series array, this sieve
        will extract the value of the time series at this position. If
        it is a real number in (0,1), the corresponding coquantile will
        be calculated first and the result will be treated as the value
        index.
        It is also possible to pass a list of integers to this argument.
        This will return the values at these positions(-1).,
        defaults to -1
    :type cut: int, optional
    :param name: Name of the object, defaults to "Last value"
    :type name: str, optional
    """
    def __init__(self,
                 cut: int = -1,
                 name: str = "Last value"):
        super().__init__(name)
        self._cut = cut if isinstance(cut, list) else [cut]

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :returns: number of features per time series
        :rtype: int
        """
        return len(self._cut)

    def sieve(self, X: np.ndarray):
        """Returns `np.array([X[i, c] for i in range(len(X))])`, where
        `c` is based on the specified `cut`.

        :param X: (onedimensional) time series dataset
        :type X: np.ndarray
        :returns: feature array (one feature for each time series)
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            prereq = self._prerequisites().get(np.expand_dims(X, axis=1))
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            for j in range(len(self._cut)):
                cut = self._cut[j]
                if 0 < cut < 1:
                    cut = np.sum((prereq[i] <= (prereq[i, -1] * cut)))
                    if cut == 0:
                        cut = 1
                elif not isinstance(cut, int):
                    raise TypeError("Cut has to be a float in (0,1) or an " +
                                    "integer")
                elif cut < 0:
                    cut = X.shape[1]
                elif cut > X.shape[1]:
                    raise IndexError("Cutting index out of range")
                result[i, j] = X[i, cut-1]
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def _prerequisites(self) -> FruitString:
        fs = FruitString()
        for c in self._cut:
            if 0 < c <  1:
                fs.preparateur = INC(zero_padding=False)
                fs.word = SimpleWord("[11]")
                break
        return fs

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: END
        """
        fs = END(self._cut, self.name)
        return fs

    def __str__(self) -> str:
        string = "END(" + \
                f"cut={self._cut})"
        return string
