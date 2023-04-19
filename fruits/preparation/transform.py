__all__ = ["INC", "STD", "NRM", "MAV", "LAG", "RIN", "JLD"]

from typing import Any, Callable, Optional, Union

import numba
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
        shift (int or float, optional): If an integer is given, the time
            series is shifted this number of indices bevor subtracting
            it from the unshifted version. In the example above, this
            shift is one. A float value will be multiplied with the
            length of the time series to get the actual shift, rounded
            up.
        zero_padding (bool, optional): If set to True, then the first
            entry in each time series will be set to 0. If False, it
            is set to the first value of the original time series.
            Defaults to True.
        overwrite (bool, optional): When set to false, the increments
            will get added as a new dimension to each time series
            instead of replacing them. This will be done for each
            dimension of the original series. Defaults to true.
    """

    def __init__(
        self,
        shift: Union[int, float] = 1,
        zero_padding: bool = True,
        overwrite: bool = True,
    ) -> None:
        self._shift = shift
        self._zero_padding = zero_padding
        self._overwrite = overwrite

    @property
    def requires_fitting(self) -> bool:
        return False

    def _transform(self, X: np.ndarray) -> np.ndarray:
        out = _increments(
            X,
            self._shift if isinstance(self._shift, int) else (
                np.ceil(self._shift * X.shape[2])
            )
        )
        if not self._zero_padding:
            out[:, :, :self._shift] = X[:, :, :self._shift]
        if self._overwrite:
            return out
        result = np.zeros((X.shape[0], 2*X.shape[1], X.shape[2]))
        result[:, :X.shape[1], :] = X
        result[:, X.shape[1]:, :] = out
        return result

    def _copy(self) -> "INC":
        return INC(self._shift, self._zero_padding, self._overwrite)

    def __eq__(self, other) -> bool:
        if (isinstance(other, INC)
                and self._zero_padding == other._zero_padding
                and self._shift == other._shift
                and self._overwrite == self._overwrite):
            return True
        return False

    def __str__(self) -> str:
        return f"INC({self._shift}, {self._zero_padding}, {self._overwrite})"


class STD(Preparateur):
    """Preparateur: Standardization

    Used for standardizing a given time series dataset.

    Args:
        separately (bool, optional): If set to true, each time series
            in the dataset will be standardized on its own. Otherwise,
            the transformation returns ``(X-mu)/std`` where ``mu`` and
            ``std`` are calculated in :meth:`STD.fit`. Defaults to true.
        var (bool, optional): Whether to standardize the variance of the
            time series. If set to false, the resulting time series will
            only be centered to zero. Defaults to True.
        dim (int, optional): If an index of a dimension in the input
            time series is given, only this dimension will be
            standardized. This only works for ``separately=True``.
            Defaults to all dimensions being standardized.
    """

    def __init__(
        self,
        separately: bool = True,
        var: bool = True,
        dim: Optional[int] = None,
    ) -> None:
        self._separately = separately
        self._div_std = var
        self._dim = dim
        self._mean = None
        self._std = None

    def _fit(self, X: np.ndarray) -> None:
        if not self._separately:
            self._mean = np.mean(X)
            if self._div_std:
                self._std = np.std(X)
            else:
                self._std = 1

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self._separately:
            if self._mean is None or self._std is None:
                raise RuntimeError("Missing call of self.fit()")
            out = (X - self._mean) / self._std
        else:
            if self._dim is None:
                mean_ = np.mean(X, axis=2)[:, :, np.newaxis]
                std_ = 1
                if self._div_std:
                    std_ = np.std(X, axis=2)[:, :, np.newaxis]
                out = (X - mean_) / std_
            else:
                mean_ = np.mean(X[:, self._dim, :], axis=1)[:, np.newaxis]
                std_ = 1
                if self._div_std:
                    std_ = np.std(X[:, self._dim, :], axis=1)[:, np.newaxis]
                out = X.copy()
                out[:, self._dim, :] = (X[:, self._dim, :] - mean_) / std_
        return out

    def _copy(self) -> "STD":
        return STD(self._separately, self._div_std, self._dim)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, STD):
            return False
        if (self._separately == other._separately and
                self._div_std == other._div_std and
                self._dim == other._dim):
            return True
        return False

    def __str__(self) -> str:
        return f"STD({self._separately}, {self._div_std}, {self._dim})"


class NRM(Preparateur):
    """Preparateur: Normalization

    Used for normalization of a given time series dataset. The
    transformation returns ``(X-min)/(max-min)`` where ``max``, ``min``
    is the maximum and minimum of single time series dimensions ``X``.
    """

    @property
    def requires_fitting(self) -> bool:
        return False

    def _transform(self, X: np.ndarray) -> np.ndarray:
        min_ = np.min(X, axis=2)[:, :, np.newaxis]
        max_ = np.max(X, axis=2)[:, :, np.newaxis]
        return (X - min_) / (max_ - min_)

    def _copy(self) -> "NRM":
        return NRM()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NRM):
            return False
        return True

    def __str__(self) -> str:
        return "NRM()"


class MAV(Preparateur):
    """Preparateur: Moving Average

    Applies a moving average to the given time series dataset.

    Args:
        width (int or float, optional): Window width for the moving
            average. This is either a float that will be multiplied by
            the length of the time series or an integer. Defaults to
            ``5``.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], int32)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(X: np.ndarray, width: int) -> np.ndarray:
        result = np.zeros_like(X)
        for i in numba.prange(X.shape[0]):
            for j in numba.prange(X.shape[1]):
                for k in range(width, X.shape[2]+1):
                    result[i, j, k-1] = np.sum(X[i, j, k-width:k]) / width
        return result

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
        return MAV._backend(X, self._w)

    def _copy(self) -> "MAV":
        return MAV(self._w_given)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MAV):
            return False
        return self._w_given == other._w_given

    def __str__(self) -> str:
        return f"MAV({self._w_given})"


class LAG(Preparateur):
    """Preparateur: Lead-Lag transform

    This preparateur applies the so called lead-lag transform to every
    dimension of the given time series.
    For one dimension ``[x_1,x_2,...,x_n]`` this results in a new series
    of two-dimensional vectors
    ``[(x_1,x_1), (x_2,x_1), (x_2,x_2), (x_3,x_2), ..., (x_n,x_n)]``.
    """

    @property
    def requires_fitting(self) -> bool:
        return False

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


class RIN(Preparateur):
    """Preparateur: Random Increments

    Calculates random increments over multiple time steps. This is a
    special case of a weighted moving average. For a random kernel
    ```
        [k_1, ..., k_w]
    ```
    drawn at a :meth:`RIN.fit` call, each dimension of the input time
    series will be transformed according to
    ```
        y_i = x_i - (k_w*x_{i-1} + ... + k_1*x_{i-w}).
    ```
    Here, `w` is the width of the window or length of the kernel.

    Args:
        width (int or callable, optional): Kernel length for the random
            gaussian distributed weights. If the kernel is set to be
            longer than the time series given in a call of :meth:`fit`,
            then it will be shortened to ``l-1`` where ``l`` is the
            length of the time series. Also a function can be supplied
            that takes an integer as input, which will be the length of
            the time series, and outputs an integer that is used as the
            kernel width. Defaults to 1.
        adaptive_width (bool, optional): When set to true, a truncated
            version of the kernel is used when the kernel normally would
            have values outside the range of the input time series. This
            is equivalent to padding ``width`` zeros at the start of the
            time series and doing a normal 1d-convolution. Defaults to
            True.
        force_positive (bool, optional): When set to true, forces all
            kernel weights to be non-negative. Defaults to false.
        overwrite (bool, optional): When set to false, the increments
            will get added as a new dimension to each time series
            instead of replacing them. This will be done for each
            dimension of the original series. Defaults to true.
    """

    @staticmethod
    @numba.njit(
        "float64[:,:,:](float64[:,:,:], float64[:])",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        w = kernel.size
        result = np.zeros(X.shape)
        for i in numba.prange(X.shape[0]):
            for j in numba.prange(X.shape[1]):
                for k in range(w, X.shape[2]):
                    s = 0
                    for l in range(k-w, k):
                        s += X[i, j, l] * kernel[l-k+w]
                    result[i, j, k] = X[i, j, k] - s
        return result

    def __init__(
        self,
        width: Union[int, Callable[[int], int]] = 1,
        adaptive_width: bool = True,
        force_positive: bool = False,
        overwrite: bool = True,
    ) -> None:
        self._width = width
        self._adaptive_width = adaptive_width
        self._force_positive = force_positive
        self._overwrite = overwrite

    def _fit(self, X: np.ndarray) -> None:
        if callable(self._width):
            self._kernel = np.random.randn(self._width(X.shape[2]))
        else:
            self._kernel = np.random.randn(min(self._width, X.shape[2]-1))
        if self._force_positive:
            for i in range(self._kernel.size):
                if self._kernel[i] < 0:
                    self._kernel[i] = -self._kernel[i]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_kernel"):
            raise RuntimeError("RIN preparateur misses a .fit() call")
        if not self._adaptive_width:
            out = RIN._backend(X, self._kernel)
        else:
            out = RIN._backend(
                np.pad(X, ((0, 0), (0, 0), (self._kernel.size, 0))),
                self._kernel,
            )
            out = out[:, :, self._kernel.size:]
        if self._overwrite:
            return out
        result = np.zeros((X.shape[0], 2*X.shape[1], X.shape[2]))
        result[:, :X.shape[1], :] = X
        result[:, X.shape[1]:, :] = out
        return result

    def _copy(self) -> "RIN":
        return RIN(
            self._width,
            self._adaptive_width,
            self._force_positive,
            self._overwrite,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RIN):
            return False
        if (self._width == other._width
            and self._adaptive_width == other._adaptive_width
            and self._force_positive == other._force_positive
            and self._overwrite == other._overwrite):
            return True
        return False

    def __str__(self) -> str:
        return (
            f"RIN({self._width}, {self._adaptive_width}, "
            f"{self._force_positive}, {self._overwrite})"
        )


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
        return f"JLD({self._d})"
