from typing import Union

import numpy as np

from fruits.preparation.abstract import DataPreparateur
from fruits.preparation.backend import _increments

class INC(DataPreparateur):
    """DataPreparateur: Increments

    For one dimension of a time series::

        X = [x_1, x_2, ..., x_n]

    this class produces the output::

        X_inc = [0, x_2-x_1, x_3-x_2, ..., x_n-x_{n-1}].

    :param zero_padding: If set to True, then the first entry in each
        time series will be set to 0. If False, it isn't changed at
        all., defaults to True
    :type zero_padding: bool, optional
    """
    def __init__(self,
                 zero_padding: bool = True):
        super().__init__("Increments")
        self._zero_padding = zero_padding

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the increments of all time series in ``X``.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        out = _increments(X)
        if self._zero_padding:
            out[:, :, 0] = 0
        return out

    def copy(self) -> "INC":
        """Returns a copy of this preparateur.

        :rtype: INC
        """
        dp = INC(self._zero_padding)
        return dp

    def __eq__(self, other) -> bool:
        if not isinstance(other, INC):
            return False
        if self._zero_padding == other._zero_padding:
            return True
        return False

    def __str__(self) -> str:
        string = "INC(" + \
                f"zero_padding={self._zero_padding})"
        return string

    def __repr__(self) -> str:
        return "fruits.preparation.transform.INC"


class STD(DataPreparateur):
    """DataPreparateur: Standardization

    Used for standardization of a given time series dataset.
    """
    def __init__(self):
        super().__init__("Standardization")
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray):
        """Fits the STD object to the given dataset by calculating the
        mean and standard deviation of the flattened dataset.

        :type X: np.ndarray
        """
        self._mean = np.mean(X)
        self._std = np.std(X)

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the standardized dataset ``(X-mu)/std`` where ``mu``
        and ``std`` are the parameters calculated in :meth:`STD.fit`.

        :type X: np.ndarray
        :returns: Standardized dataset.
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Missing call of self.fit()")
        out = (X - self._mean) / self._std
        return out

    def copy(self) -> "STD":
        """Returns a copy of this preparateur.

        :rtype: STD
        """
        return STD()

    def __eq__(self, other) -> bool:
        return True

    def __str__(self) -> str:
        return "STD"

    def __repr__(self) -> str:
        return "fruits.preparation.transform.STD"


class MAV(DataPreparateur):
    """DataPreparateur: Moving Average

    Applies a moving average to the given time series dataset.

    :param width: Window width for the moving average. This is either a
        float that will be multiplied by the length of the time series
        or an integer., defaults to 5
    :type width: Union[int, float], optional
    """
    def __init__(self, width: Union[int, float] = 5):
        super().__init__("Moving Average")
        if isinstance(width, float):
            if not 0.0 < width < 1.0:
                raise ValueError("If width is a float, it has to be in (0,1)")
        elif isinstance(width, int):
            if not width > 0:
                raise ValueError("If width is an integer, it has to be > 0")
        else:
            raise TypeError("width has to be an integer or a float in (0,1)")
        self._w_given = width
        self._w = None

    def fit(self, X: np.ndarray):
        """Fits the MAV preparateur to the given dataset by calculating
        the window width if needed.

        :type X: np.ndarray
        """
        if isinstance(self._w_given, float):
            self._w = int(self._w_given * X.shape[2])
            if self._w <= 0:
                self._w = 1
        else:
            self._w = self._w_given

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if self._w is None:
            raise RuntimeError("Missing call of self.fit()")
        out = np.cumsum(X, axis=2)
        out[:, :, self._w:] = out[:, :, self._w:] - out[:, :, :-self._w]
        out[:, :, (self._w-1):] = out[:, :, (self._w-1):] / self._w
        out[:, :, :(self._w-1)] = X[:, :, :(self._w-1)]
        return out

    def copy(self) -> "MAV":
        """Returns a copy of this preparateur.

        :rtype: MAV
        """
        return MAV(self._w_given)

    def __eq__(self, other) -> bool:
        return (self._w_given == other._w_given)

    def __str__(self) -> str:
        return f"MAV(width={self._w_given})"

    def __repr__(self) -> str:
        return "fruits.preparation.transform.MAV"


class LAG(DataPreparateur):
    """DataPreparateur: Lead-Lag transform

    This preparateur applies the so called lead-lag transform to every
    dimension of the given time series.
    For one dimension ``[x_1,x_2,...,x_n]`` this results in a new
    two-dimensional vector
    ``[(x_1,x_1),(x_2,x_1),(x_2,x_2),(x_3,x_2),...,(x_n,x_n)]``.
    """
    def __init__(self):
        super().__init__("Lead-Lag transform")

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        X_new = np.zeros((X.shape[0], 2 * X.shape[1], 2 * X.shape[2] - 1))
        for i in range(X.shape[1]):
            X_new[:, 2*i, 0::2] = X[:, i, :]
            X_new[:, 2*i, 1::2] = X[:, i, 1:]
            X_new[:, 2*i+1, 0::2] = X[:, i, :]
            X_new[:, 2*i+1, 1::2] = X[:, i, :-1]
        return X_new

    def copy(self) -> "LAG":
        """Returns a copy of this preparateur.

        :rtype: LAG
        """
        return LAG()

    def __eq__(self, other) -> bool:
        return True

    def __str__(self) -> str:
        return f"LAG()"

    def __repr__(self) -> str:
        return "fruits.preparation.transform.LAG"
