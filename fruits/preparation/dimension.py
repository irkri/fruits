import numpy as np

from fruits.preparation.abstract import DataPreparateur

class LAG(DataPreparateur):
    """DataPreparateur: Lag

    This preparateur adds more dimensions to each given time series.
    For every dimension, a lagged version of the dimension is appended
    to the time series. The lagged series is the original series shifted
    to right by a given number ``lag`` and cropped at the end. The first
    ``lag`` numbers are set to zero.

    :param lag: Number of indices the first element of each time series
        dimension is shifted to the right., defaults to 1
    :type lag: int, optional
    """
    def __init__(self, lag: int = 1):
        super().__init__("Lag")
        if not isinstance(lag, int) or lag <= 0:
            raise ValueError("lag has to be a integer > 0")
        self._lag = lag

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        X_new = np.zeros((X.shape[0], 2*X.shape[1], X.shape[2]))
        for i in range(X.shape[1]):
            X_new[:, 2*i, :] = X[:, i, :]
            X_new[:, 2*i+1, :] = np.roll(X[:, i, :], self._lag, axis=1)
            X_new[:, 2*i+1, :self._lag] = 0
        return X_new

    def copy(self) -> "LAG":
        """Returns a copy of this preparateur.

        :rtype: LAG
        """
        return LAG(self._lag)

    def __eq__(self, other) -> bool:
        return False

    def __str__(self) -> str:
        return f"LAG(lag={self._lag})"

    def __repr__(self) -> str:
        return "fruits.preparation.dimension.LAG"


class ONE(DataPreparateur):
    """DataPreparateur: Ones

    Preparateur that appends a dimension to each time series consisting
    of only ones.
    """
    def __init__(self):
        super().__init__("One")

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        X_new = np.ones((X.shape[0], X.shape[1]+1, X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        return X_new

    def copy(self) -> "ONE":
        """Returns a copy of this preparateur.

        :rtype: ONE
        """
        return ONE()

    def __eq__(self, other) -> bool:
        return True

    def __str__(self) -> str:
        return f"ONE()"

    def __repr__(self) -> str:
        return "fruits.preparation.dimension.ONE"


class DIM(DataPreparateur):
    """DataPreparateur: Dimension Creator
    
    Creates a new dimension in the given (multidimensional) time series
    dataset based on the supplied function.

    :param f: Function that takes in a three dimensional numpy array of
        shape ``(n, d, l)`` and returns an array of shape ``(n, p, l)``
        where ``p`` is an arbitrary integer matching the number of new
        dimensions that will be added to the input array.
    :type f: Callable
    """
    def __init__(self, f):
        super().__init__("Dimension Creator")
        self._function = f

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        new_dims = self._function(X)
        X_new = np.zeros((X.shape[0],
                          X.shape[1] + new_dims.shape[1],
                          X.shape[2]))
        X_new[:, :X.shape[1], :] = X[:, :, :]
        X_new[:, X.shape[1]:, :] = new_dims[:, :, :]
        return X_new

    def copy(self) -> "DIM":
        """Returns a copy of this preparateur.

        :rtype: DIM
        """
        return DIM(self._function)

    def __eq__(self, other) -> bool:
        return False

    def __str__(self) -> str:
        return f"DIM(f={self._function.__name__})"

    def __repr__(self) -> str:
        return "fruits.preparation.dimension.DIM"
