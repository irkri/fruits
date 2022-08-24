from typing import Any, Union

import numpy as np

from fruits._backend import _increments
from fruits.preparation.abstract import DataPreparateur


class INC(DataPreparateur):
    """DataPreparateur: Increments

    For one dimension of a time series::

        X = [x_1, x_2, ..., x_n]

    this class produces the output::

        X_inc = [0, x_2-x_1, x_3-x_2, ..., x_n-x_{n-1}].

    Args:
        zero_padding (bool, optional): If set to True, then the first
            entry in each time series will be set to 0. If False, it
            is set to the first value of the original time series.
            Defaults to True.
    """

    def __init__(self, zero_padding: bool = True) -> None:
        super().__init__("Increments")
        self._zero_padding = zero_padding

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        out = _increments(X)
        if not self._zero_padding:
            out[:, :, 0] = X[:, :, 0]
        return out

    def copy(self) -> "INC":
        dp = INC(self._zero_padding)
        return dp

    def __eq__(self, other) -> bool:
        if (isinstance(other, INC)
                and self._zero_padding == other._zero_padding):
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

    Used for standardization of a given time series dataset. The
    transformation returns ``(X-mu)/std`` where ``mu`` and ``std`` are
    the parameters calculated in :meth:`STD.fit`.
    """

    def __init__(self) -> None:
        super().__init__("Standardization")
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the STD object to the given dataset by calculating the
        mean and standard deviation of the flattened dataset.
        """
        self._mean = np.mean(X)
        self._std = np.std(X)

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("Missing call of self.fit()")
        out = (X - self._mean) / self._std
        return out

    def copy(self) -> "STD":
        return STD()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, STD):
            return False
        return True

    def __str__(self) -> str:
        return "STD"

    def __repr__(self) -> str:
        return "fruits.preparation.transform.STD"


class MAV(DataPreparateur):
    """DataPreparateur: Moving Average

    Applies a moving average to the given time series dataset.

    Args:
        width (int or float, optional): Window width for the moving
            average. This is either a float that will be multiplied by
            the length of the time series or an integer. Defaults to
            ``5``.
    """

    def __init__(self, width: Union[int, float] = 5) -> None:
        super().__init__("Moving Average")
        if isinstance(width, float):
            if not 0.0 < width < 1.0:
                raise ValueError("If width is a float, it has to be in (0,1)")
        elif isinstance(width, int):
            if width <= 0:
                raise ValueError("If width is an integer, it has to be > 0")
        else:
            raise TypeError("width has to be an integer or a float in (0,1)")
        self._w_given = width
        self._w: int

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the MAV preparateur to the given dataset by calculating
        the window width if needed.
        """
        if isinstance(self._w_given, float):
            self._w = int(self._w_given * X.shape[2])
            if self._w <= 0:
                self._w = 1
        else:
            self._w = self._w_given

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not hasattr(self, "_w"):
            raise RuntimeError("Missing call of self.fit()")
        out = np.cumsum(X, axis=2)
        out[:, :, self._w:] = out[:, :, self._w:] - out[:, :, :-self._w]
        out[:, :, (self._w-1):] = out[:, :, (self._w-1):] / self._w
        out[:, :, :(self._w-1)] = X[:, :, :(self._w-1)]
        return out

    def copy(self) -> "MAV":
        return MAV(self._w_given)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MAV):
            return False
        return self._w_given == other._w_given

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

    def __init__(self) -> None:
        super().__init__("Lead-Lag transform")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X_new = np.zeros((X.shape[0], 2 * X.shape[1], 2 * X.shape[2] - 1))
        for i in range(X.shape[1]):
            X_new[:, 2*i, 0::2] = X[:, i, :]
            X_new[:, 2*i, 1::2] = X[:, i, 1:]
            X_new[:, 2*i+1, 0::2] = X[:, i, :]
            X_new[:, 2*i+1, 1::2] = X[:, i, :-1]
        return X_new

    def copy(self) -> "LAG":
        return LAG()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LAG):
            return False
        return True

    def __str__(self) -> str:
        return "LAG()"

    def __repr__(self) -> str:
        return "fruits.preparation.transform.LAG"
