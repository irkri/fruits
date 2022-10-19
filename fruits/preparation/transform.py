__all__ = ["INC", "STD", "MAV", "LAG", "JLD"]

from typing import Any, Union

import numpy as np

from ..cache import _increments
from .abstract import Preparateur


class INC(Preparateur):
    """Preparateur: Increments

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
        self._zero_padding = zero_padding

    def _transform(self, X: np.ndarray) -> np.ndarray:
        out = _increments(X)
        if not self._zero_padding:
            out[:, :, 0] = X[:, :, 0]
        return out

    def _copy(self) -> "INC":
        return INC(self._zero_padding)

    def __eq__(self, other) -> bool:
        if (isinstance(other, INC)
                and self._zero_padding == other._zero_padding):
            return True
        return False

    def __str__(self) -> str:
        return f"INC(zero_padding={self._zero_padding})"


class STD(Preparateur):
    """Preparateur: Standardization

    Used for standardization of a given time series dataset. The
    transformation returns ``(X-mu)/std`` where ``mu`` and ``std`` are
    the parameters calculated in :meth:`STD.fit`.
    """

    def __init__(self) -> None:
        self._mean = None
        self._std = None

    def _fit(self, X: np.ndarray) -> None:
        self._mean = np.mean(X)
        self._std = np.std(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("Missing call of self.fit()")
        out = (X - self._mean) / self._std
        return out

    def _copy(self) -> "STD":
        return STD()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, STD):
            return False
        return True

    def __str__(self) -> str:
        return "STD"


class MAV(Preparateur):
    """Preparateur: Moving Average

    Applies a moving average to the given time series dataset.

    Args:
        width (int or float, optional): Window width for the moving
            average. This is either a float that will be multiplied by
            the length of the time series or an integer. Defaults to
            ``5``.
    """

    def __init__(self, width: Union[int, float] = 5) -> None:
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

    def _fit(self, X: np.ndarray) -> None:
        if isinstance(self._w_given, float):
            self._w = int(self._w_given * X.shape[2])
            if self._w <= 0:
                self._w = 1
        else:
            self._w = self._w_given

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_w"):
            raise RuntimeError("Missing call of self.fit()")
        out = np.cumsum(X, axis=2)
        out[:, :, self._w:] = (
            out[:, :, self._w:] - out[:, :, :-self._w]  # type: ignore
        )
        out[:, :, (self._w-1):] = out[:, :, (self._w-1):] / self._w
        out[:, :, :(self._w-1)] = X[:, :, :(self._w-1)]
        return out

    def _copy(self) -> "MAV":
        return MAV(self._w_given)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MAV):
            return False
        return self._w_given == other._w_given

    def __str__(self) -> str:
        return f"MAV(width={self._w_given})"


class LAG(Preparateur):
    """Preparateur: Lead-Lag transform

    This preparateur applies the so called lead-lag transform to every
    dimension of the given time series.
    For one dimension ``[x_1,x_2,...,x_n]`` this results in a new series
    of two-dimensional vectors
    ``[(x_1,x_1), (x_2,x_1), (x_2,x_2), (x_3,x_2), ..., (x_n,x_n)]``.
    """

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_new = np.zeros((X.shape[0], 2 * X.shape[1], 2 * X.shape[2] - 1))
        for i in range(X.shape[1]):
            X_new[:, 2*i, 0::2] = X[:, i, :]
            X_new[:, 2*i, 1::2] = X[:, i, 1:]
            X_new[:, 2*i+1, 0::2] = X[:, i, :]
            X_new[:, 2*i+1, 1::2] = X[:, i, :-1]
        return X_new

    def _copy(self) -> "LAG":
        return LAG()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LAG):
            return False
        return True

    def __str__(self) -> str:
        return "LAG()"


class JLD(Preparateur):
    """Preparatuer: Johnson-Lindenstrauss Dimensionality Reduction

    This preparateur transforms the input dimensions of one time series
    by multiplying each time step with a vector having random gaussian
    distributed entries.
    According to the Johnson-Lindenstrauss lemma, the distance between
    vectors in the lower dimensional space are nearly preserved.

    Args:
        dim (int or float, optional): The number of output dimensions.
            If a float ``f`` in (0, 1) is given, this number will be the
            smallest integer ``>= 24*log(d) / (3*f**2 - 2*f**3)``, where
            ``d`` is the number of input dimensions. Defaults to
            ``0.99``. The default argument should only be used when
            dealing with high dimensional time series (``d>500``). It is
            designed so that the Johnson-Lindenstrauss lemma is
            applicable.
    """

    def __init__(self, dim: Union[int, float] = 0.99) -> None:
        if isinstance(dim, float) and not (0 < dim < 1):
            raise ValueError(
                "'dim' has to be an integer or a float in (0, 1)"
            )
        self._d = dim
        self._operator: np.ndarray

    def _fit(self, X: np.ndarray) -> None:
        if isinstance(self._d, float):
            div_ = 3*self._d**2 - 2*self._d**3
            d = int(24 * np.log(X.shape[1]) / div_) + 1
        else:
            d = self._d
        self._operator = np.random.randn(d, X.shape[1])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(self._operator, X)

    def _copy(self) -> "JLD":
        return JLD()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JLD):
            return False
        if self._d == other._d:
            return True
        return False

    def __str__(self) -> str:
        return f"JLD(dim={self._d})"
