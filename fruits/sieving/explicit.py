import numpy as np

from fruits.cache import FruitString
from fruits.core.wording import SimpleWord
from fruits.sieving.abstract import FeatureSieve
from fruits.preparation.transform import INC

class MAX(FeatureSieve):
    """FeatureSieve: Maximal value
    
    This sieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.

    :param cut: If ``cut`` is an index of the time series array, the
        sieve searches for the maximum in ``X[:cut]``. If it is a real
        number in (0,1), the corresponding 'coquantile' will be
        calculated first. This option can also be a list of floats or
        integers which will be treated in the same way., defaults to -1
    :type cut: int/float or list of integers/floats, optional
    :param segments: If set to ``True``, then the cutting indices will 
        be sorted and treated as interval borders and the maximum in
        each interval will be sieved. The left interval border is
        reduced by 1 before slicing. This means that an input of
        ``cut=[1,5,10]`` results in two features ``max(X[0:5])`` and
        ``max(X[4:10])``.
        If set to ``False``, then the left interval border is always 0.,
        defaults to ``False``
    :type segments: bool, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False):
        super().__init__("Maximal value")
        self._cut = cut if isinstance(cut, list) else [cut]
        if segments and len(self._cut) == 1:
            raise ValueError("If 'segments' is set to True, then 'cut'"+
                             " has to be a list length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this sieve produces.
        
        :rtype: int
        """
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.
        
        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            pq = self._prerequisites()
            pq.process(np.expand_dims(X, axis=1))
            prereq = pq.get()
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

    def summary(self) -> str:
        string = f"MAX"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: MAX
        """
        fs = MAX(self._cut, self._segments)
        return fs

    def __str__(self) -> str:
        string = "MAX(" + \
                f"cut={self._cut}, " + \
                f"segments={self._segments})"
        return string


class MIN(FeatureSieve):
    """FeatureSieve: Minimal value
    
    This sieve returns the minimal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.

    :param cut: If ``cut`` is an index of the time series array, the
        sieve searches for the minimum in ``X[:cut]``. If it is a real
        number in (0,1), the corresponding 'coquantile' will be
        calculated first. This option can also be a list of floats or
        integers which will be treated in the same way., defaults to -1
    :type cut: int/float or list of integers/floats, optional
    :param segments: If set to ``True``, then the cutting indices will 
        be sorted and treated as interval borders and the minimum in
        each interval will be sieved. The left interval border is
        reduced by 1 before slicing. This means that an input of
        ``cut=[1,5,10]`` results in two features ``min(X[0:5])`` and
        ``min(X[5:10])``.
        If set to ``False``, then the left interval border is always 0.,
        defaults to ``False``
    :type segments: bool, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False):
        super().__init__("Minimum value")
        self._cut = cut if isinstance(cut, list) else [cut]
        if segments and len(self._cut) == 1:
            raise ValueError("If 'segments' is set to True, then 'cut'"+
                             " has to be a list length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :rtype: int
        """
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.
        
        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            pq = self._prerequisites()
            pq.process(np.expand_dims(X, axis=1))
            prereq = pq.get()
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

    def summary(self) -> str:
        string = f"MIN"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: MIN
        """
        fs = MIN(self._cut, self._segments)
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
    
    :param cut: If ``cut`` is an index of the time series array, the
        sieve returns ``X[cut-1]``. If it is a real number in (0,1), the
        corresponding 'coquantile' will be calculated first. This option
        can also be a list of floats or integers which will be treated
        in the same way., defaults to -1
    :type cut: int/float or list of integers/floats, optional
    """
    def __init__(self,
                 cut: int = -1):
        super().__init__("Last value")
        self._cut = cut if isinstance(cut, list) else [cut]

    def nfeatures(self) -> int:
        """Returns the number of features this FeatureSieve produces.
        
        :returns: number of features per time series
        :rtype: int
        """
        return len(self._cut)

    def sieve(self, X: np.ndarray):
        """Returns the transformed data. See the class definition for
        detailed information.
        
        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        if self._prereqs is not None:
            prereq = self._prereqs.get()
        else:
            pq = self._prerequisites()
            pq.process(np.expand_dims(X, axis=1))
            prereq = pq.get()
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

    def summary(self) -> str:
        string = f"END -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self):
        """Returns a copy of this object.
        
        :rtype: END
        """
        fs = END(self._cut)
        return fs

    def __str__(self) -> str:
        string = "END(" + \
                f"cut={self._cut})"
        return string
