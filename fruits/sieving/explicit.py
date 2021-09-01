from abc import abstractmethod

import numpy as np

from fruits.sieving.abstract import FeatureSieve

class ExplicitSieve(FeatureSieve):
    """Abstract class that has the ability to calculate cutting points
    as indices in the time series based on a given 'coquantile'.
    A (non-scaled) value returned by an explicit sieve always also is a
    value in the original time series.

    :param cut: If ``cut`` is an index in the time series array, the
        features are sieved from ``X[:cut]``. If it is a real number in
        ``(0,1)``, the corresponding 'coquantile' will be calculated
        first. This option can also be a list of floats or integers
        which will be treated the same way., defaults to -1
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
                 segments: bool = False,
                 name: str = "Abstract Explicit Sieve"):
        super().__init__(name)
        self._cut = cut if isinstance(cut, list) else [cut]
        for c in self._cut:
            if 0 < c < 1:
                self._requisite = "INC -> [11]"
            elif not (c == -1 or (c > 0 and isinstance(c, int))):
                raise ValueError("Unsupported input for option 'cut'")
        if segments and len(self._cut) == 1:
            raise ValueError("If 'segments' is set to True, then 'cut'"+
                             " has to be a list of length >= 2.")
        self._segments = segments

    def _transform_cuts(self, X: np.ndarray, req: np.ndarray) -> list:
        # transforms the input cuts based on the given time series
        new_cuts = []
        for j in range(len(self._cut)):
            cut = self._cut[j]
            if 0 < cut < 1:
                cut = np.sum((req <= (req[-1] * cut)))
                if cut == 0:
                    cut = 1
            elif not isinstance(cut, int):
                raise TypeError("Cut has to be a float in (0,1) or an " +
                                "integer")
            elif cut == -1:
                cut = X.shape[0]
            elif cut > X.shape[0]:
                raise IndexError("Cutting index out of range")
            new_cuts.append(cut)
        if self._segments:
            new_cuts = sorted(list(new_cuts))
        return new_cuts

    def nfeatures(self) -> int:
        """Returns the number of features this sieve produces.

        :rtype: int
        """
        if self._segments:
            return len(self._cut) - 1
        else:
            return len(self._cut)

    @abstractmethod
    def sieve(self, X: np.ndarray) -> np.ndarray:
        pass


class MAX(ExplicitSieve):
    """FeatureSieve: Maximal value

    This sieve returns the maximal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.

    :type cut: int/float or list of integers/floats, optional
    :type segments: bool, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False):
        super().__init__(cut, segments, "Maximal value")

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        req = self._get_requisite(X)[:, 0, :]
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            new_cuts = self._transform_cuts(X[i], req[i])
            if self._segments:
                for j in range(1, len(new_cuts)):
                    result[i, j-1] = np.max(X[i, new_cuts[j-1]-1:new_cuts[j]])
            else:
                for j in range(len(new_cuts)):
                    result[i, j] = np.max(X[i, :new_cuts[j]])
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        string = f"MAX"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "MAX":
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


class MIN(ExplicitSieve):
    """FeatureSieve: Minimal value

    This sieve returns the minimal value for each slice of a time
    series in a given dataset. The slices are determined by the option
    ``cut``.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.

    :type cut: int/float or list of integers/floats, optional
    :type segments: bool, optional
    """
    def __init__(self,
                 cut: int = -1,
                 segments: bool = False):
        super().__init__(cut, segments, "Minimum value")

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        req = self._get_requisite(X)[:, 0, :]
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            new_cuts = self._transform_cuts(X[i], req[i])
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

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        string = f"MIN"
        if self._segments:
            string += " [segments]"
        string += f" -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "MIN":
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


class END(ExplicitSieve):
    """FeatureSieve: Last value

    This FeatureSieve returns the last value of each time series in a
    given dataset.
    For more information on the available arguments, have a look at the
    definition of :class:`~fruits.sieving.explicit.ExplicitSieve`.
    The option 'segments' will be ignored in this sieve.

    :type cut: int/float or list of integers/floats, optional
    """
    def __init__(self,
                 cut: int = -1):
        super().__init__(cut, False, "Last value")

    def sieve(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        """
        req = self._get_requisite(X)[:, 0, :]
        result = np.zeros((X.shape[0], self.nfeatures()))
        for i in range(X.shape[0]):
            new_cuts = self._transform_cuts(X[i], req[i])
            for j in range(len(new_cuts)):
                result[i, j] = X[i, new_cuts[j]-1]
        if self.nfeatures() == 1:
            return result[:, 0]
        return result

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        string = f"END -> {self.nfeatures()}:"
        for x in self._cut:
            string += f"\n   > {x}"
        return string

    def copy(self) -> "END":
        """Returns a copy of this object.

        :rtype: END
        """
        fs = END(self._cut)
        return fs

    def __str__(self) -> str:
        string = "END(" + \
                f"cut={self._cut})"
        return string
