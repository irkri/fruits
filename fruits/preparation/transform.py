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
            raise RuntimeError("Missing call of fit method")
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
