__all__ = [
    "INC", "STD", "NRM", "MAV", "LAG", "FFN", "RIN", "RDW", "JLD", "FUN",
]

from typing import Any, Callable, Literal, Union

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
    """

    def __init__(
        self,
        shift: Union[int, float] = 1,
        zero_padding: bool = True,
    ) -> None:
        self._shift = shift
        self._zero_padding = zero_padding

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
        return out

    def _copy(self) -> "INC":
        return INC(self._shift, self._zero_padding)

    def __eq__(self, other) -> bool:
        if (isinstance(other, INC)
                and self._zero_padding == other._zero_padding
                and self._shift == other._shift):
            return True
        return False

    def __str__(self) -> str:
        return f"INC({self._shift}, {self._zero_padding})"


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
    """

    def __init__(
        self,
        separately: bool = True,
        var: bool = True,
    ) -> None:
        self._separately = separately
        self._div_std = var
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
            mean_ = np.mean(X, axis=2)[:, :, np.newaxis]
            std_ = np.ones((X.shape[0], X.shape[1], 1))
            if self._div_std:
                std_ = np.std(X, axis=2)[:, :, np.newaxis]
            out = X - mean_
            out = np.where(std_ == 0, out, out / std_)
        return out

    def _copy(self) -> "STD":
        return STD(self._separately, self._div_std)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, STD):
            return False
        if (self._separately == other._separately and
                self._div_std == other._div_std):
            return True
        return False

    def __str__(self) -> str:
        return f"STD({self._separately}, {self._div_std})"


class NRM(Preparateur):
    """Preparateur: Normalization

    Used for normalization of a given time series dataset. The
    transformation returns ``(X-min)/(max-min)`` where ``max``, ``min``
    is the maximum and minimum of single time series dimensions ``X``.

    Args:
        scale_dim (bool, optional): If set to false, each dimensions is
            scaled independently from all other dimensions to [0,1] in
            each time series. If set to true, ``max`` and ``min`` are
            evaluated over all dimensions and all time steps. This way,
            the difference in magnitudes of dimensions is preserved.
            Defaults to false.
    """

    def __init__(self, scale_dim: bool = False) -> None:
        self._scale_dim = scale_dim

    @property
    def requires_fitting(self) -> bool:
        return False

    def _transform(self, X: np.ndarray) -> np.ndarray:
        min_ = np.min(X, axis=2)
        max_ = np.max(X, axis=2)
        if self._scale_dim:
            min_ = np.min(min_, axis=1)[:, np.newaxis]
            max_ = np.max(max_, axis=1)[:, np.newaxis]
        mask = (min_ != max_)
        if self._scale_dim:
            mask = mask[:, 0]
        min_ = min_[mask][:, np.newaxis]
        max_ = max_[mask][:, np.newaxis]
        out = np.zeros_like(X)
        out[mask] = (X[mask] - min_) / (max_ - min_)
        out[~mask] = 0
        return out

    def _copy(self) -> "NRM":
        return NRM(scale_dim=self._scale_dim)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NRM) and self._scale_dim == other._scale_dim:
            return True
        return False

    def __str__(self) -> str:
        return f"NRM({self._scale_dim})"


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


class FFN(Preparateur):
    """Preparateur: Feed-Forward Two-Layer Neural Network

    Transforms single values of a time series. A neural network with one
    hidden layer and a ReLU activation function is used. All weights and
    biases are gaussian distributed with mean zero.

    Args:
        d_hidden (int, optional): Number of nodes in the hidden layer.
            Defaults to 10.
        center (bool, optional): If set to true, each time series will
            be explicitly centered before transforming with a bias drawn
            from a normal distribution with mean 0. If set to false, all
            time series are transformed without prior centering and the
            bias will be drawn from a normal distribution with a mean
            equal to the estimated mean of the training data calculated
            in ``FFN.fit(X_train)``. Defaults to false.
        relu_out (bool, optional): Whether to use a ReLU activation on
            the output too. Defaults to false.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], f8[:], f8[:], f8[:], b1)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(
        X: np.ndarray,
        weights1: np.ndarray,
        biases: np.ndarray,
        weights2: np.ndarray,
        relu_out: bool,
    ) -> np.ndarray:
        result = np.zeros(X.shape)
        for i in numba.prange(X.shape[0]):
            for j in numba.prange(X.shape[1]):
                for k in numba.prange(X.shape[2]):
                    layer1 = weights1 * (X[i, j, k] - biases)
                    layer1 = (layer1 * (layer1 > 0))
                    layer2 = np.sum(weights2 * layer1)
                    if relu_out:
                        layer2 = (layer2 * (layer2 > 0))
                    result[i, j, k] = layer2
        return result

    def __init__(
        self,
        d_hidden: int = 10,
        center: bool = False,
        relu_out: bool = False,
    ) -> None:
        self._d_hidden = d_hidden
        self._center = center
        self._relu_out = relu_out

    def _fit(self, X: np.ndarray) -> None:
        self._weights1 = np.random.normal(scale=1.0, size=self._d_hidden)
        self._biases = np.random.normal(
            loc=0 if self._center else np.mean(X),
            scale=np.std(X),
            size=self._d_hidden,
        )
        self._weights2 = np.random.normal(scale=1.0, size=self._d_hidden)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_weights1"):
            raise RuntimeError("FFN was not fitted")
        X_in = X
        if self._center:
            X_in = X - np.mean(X, axis=2)[:, :, np.newaxis]
        return FFN._backend(
            X_in,
            self._weights1,
            self._biases,
            self._weights2,
            self._relu_out,
        )

    def _copy(self) -> "FFN":
        return FFN(
            d_hidden=self._d_hidden,
            center=self._center,
            relu_out=self._relu_out,
        )

    def __str__(self) -> str:
        return f"FFN({self._d_hidden}, {self._center}, {self._relu_out})"


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
            false.
        out_dim (int, optional): The number of output dimensions. Any
            integer less than or equal to the number of input dimensions
            is allowed. For each output dimension an (approximately)
            equal number of input dimensions are convolved with a random
            2D kernel. Defaults to -1, which corresponds to
            ``out_dim=in_dim``.
        force_sum_one (bool, optional): When set to true, a uniform
            distribution on ``[-1, 1]`` is used to sample kernel
            weights. After sampling, the values will be forced to sum up
            to one while keeping interval borders. Sampled values close
            to zero will therefore be changed a lot more then values
            close to -1 or 1. If set to false, the kernel weights are
            sampled from a standard normal distribution and centered
            again after sampling to ensure mean zero. Defaults to false.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], f8[:,:], i4[:], i4[:])",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(
        X: np.ndarray,
        kernel: np.ndarray,
        ndim: np.ndarray,
        dims: np.ndarray,
    ) -> np.ndarray:
        w = kernel.shape[1]
        result = np.zeros((X.shape[0], ndim.size, X.shape[2]))
        for i in numba.prange(X.shape[0]):
            start_dim = 0
            end_dim = 0
            for new_dim in range(ndim.size):
                end_dim += ndim[new_dim]
                for k in range(w, X.shape[2]):
                    s = 0
                    for j in range(start_dim, end_dim):
                        for l in range(k-w, k):
                            s += - X[i, dims[j], l] * kernel[j, l-k+w]
                        s += X[i, j, k]
                    result[i, new_dim, k] = s
                start_dim += ndim[new_dim]
        return result

    def __init__(
        self,
        width: Union[int, Callable[[int], int]] = 1,
        adaptive_width: bool = False,
        out_dim: int = -1,
        force_sum_one: bool = False,
    ) -> None:
        self._width = width
        self._adaptive_width = adaptive_width
        self._out_dim = out_dim
        self._force_sum_one = force_sum_one

    def _fit(self, X: np.ndarray) -> None:
        width = self._width(X.shape[2]) if callable(self._width) else (
            min(self._width, X.shape[2]-1)
        )
        out_dim = self._out_dim if self._out_dim > 0 else X.shape[1]
        if out_dim > X.shape[1]:
            raise ValueError(
                f"Output dimensions ({out_dim}) should be "
                f"<= input dimensions ({X.shape[1]})"
            )
        quotient, remainder = divmod(X.shape[1], out_dim)
        self._ndim_per_kernel = np.array(
            [quotient + 1] * remainder + [quotient] * (out_dim - remainder),
            dtype=np.int32,
        )
        self._dims_per_kernel = np.random.choice(
            X.shape[1], size=X.shape[1], replace=False,
        ).astype(np.int32)
        if self._force_sum_one:
            while True:
                self._kernel = np.random.uniform(
                    -1., 1.,
                    size=(X.shape[1], width),
                )
                change = 1.0 - np.sum(self._kernel, axis=1)
                diff = 1.0 - np.abs(self._kernel)
                diffsum = np.sum(diff, axis=1)
                if np.sum(diffsum < 1e-5) > 0:
                    continue
                self._kernel += diff * (change / diffsum)[:, np.newaxis]
                break
        else:
            self._kernel = np.random.normal(size=(X.shape[1], width))
            self._kernel -= np.mean(self._kernel, axis=1)[:, np.newaxis]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_kernel"):
            raise RuntimeError("RIN preparateur misses a .fit() call")

        if not self._adaptive_width:
            out = RIN._backend(
                X,
                self._kernel,
                self._ndim_per_kernel,
                self._dims_per_kernel,
            )
        else:
            out = RIN._backend(
                np.pad(X, ((0, 0), (0, 0), (self._kernel.shape[1], 0))),
                self._kernel,
                self._ndim_per_kernel,
                self._dims_per_kernel,
            )
            out = out[:, :, self._kernel.shape[1]:]
        return out

    def _copy(self) -> "RIN":
        return RIN(
            width=self._width,
            adaptive_width=self._adaptive_width,
            out_dim=self._out_dim,
            force_sum_one=self._force_sum_one,
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RIN) and (self._width == other._width
                and self._adaptive_width == other._adaptive_width
                and self._out_dim == other._out_dim
                and self._force_sum_one == other._force_sum_one):
            return True
        return False

    def __str__(self) -> str:
        return (
            f"RIN({self._width}, {self._adaptive_width}, "
            f"{self._out_dim}, {self._force_sum_one})"
        )


class RDW(Preparateur):
    """Preparatuer: Random Dimension Weights

    This preparateur exponentially scales dimensions in the time series
    by a random exponent uniformly for all time steps.

    Args:
        dist ('dirichlet' or 'uniform'): Type of distribution used for
            drawing the exponents. The parameters for a dirichlet
            distribution are chosen proportional to the maximal absolute
            value of each dimension in the training set. Defaults to
            'dirichlet'.
    """
    def __init__(
        self,
        dist: Literal["dirichlet", "uniform"] = "dirichlet",
    ) -> None:
        self._dist: Literal["dirichlet", "uniform"] = dist

    def _fit(self, X: np.ndarray) -> None:
        if self._dist == "dirichlet":
            alphas = np.max(np.mean(np.abs(X), axis=0), axis=1)
            alphas[alphas!=0] = alphas[alphas!=0] / np.max(alphas[alphas!=0])
            if np.sum(alphas == 0) >= 1:
                alphas += 1e-5
            self._weights = np.random.dirichlet(alphas)
        else:
            self._weights = np.random.random(X.shape[1])
            self._weights = self._weights / np.sum(self._weights)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X ** self._weights[np.newaxis, :, np.newaxis]

    def _copy(self) -> "RDW":
        return RDW(self._dist)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RDW) and other._dist == self._dist:
            return True
        return False

    def __str__(self) -> str:
        return f"RDW({self._dist!r})"


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
        return JLD(dim=self._d)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JLD):
            return False
        if self._d == other._d:
            return True
        return False

    def __str__(self) -> str:
        return f"JLD({self._d})"


class FUN(Preparateur):
    """Preparatuer: Function Transform

    This preparateur transforms the input time series dataset by
    applying the given function.

    Args:
        f (callable): A function that takes in a three dimensional numpy
            array containing a time series dataset with shape
            ``(n_samples, n_dimensions, length)`` and returns a
            transformed dataset of the same type.
    """

    def __init__(self, f: Callable[[np.ndarray], np.ndarray]) -> None:
        self._function = f

    @property
    def requires_fitting(self) -> bool:
        return False

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._function(X)

    def _copy(self) -> "FUN":
        return FUN(self._function)

    def __eq__(self, other: Any) -> bool:
        return False

    def __str__(self) -> str:
        return f"FUN({self._function})"
