__all__ = [
    "INC", "STD", "NRM", "MAV", "LAG", "FFN", "RIN", "RDW", "JLD", "SPE",
    "RPE", "FUN",
]

from typing import Any, Callable, Literal, Optional, Union

import numba
import numpy as np

from ..cache import CacheType, _increments
from .abstract import Preparateur


class INC(Preparateur):
    """Preparateur: Increments

    For one dimension of a time series::

        X = [x_1, x_2, ..., x_n]

    this class produces the output::

        X_inc = [0, x_2-x_1, x_3-x_2, ..., x_n-x_{n-1}].

    Args:
        shift (int, float or callable, optional): If an integer is
            given, the time series is shifted this number of indices
            bevor subtracting it from the unshifted version. In the
            example above, this shift is one. A float value will be
            multiplied with the length of the time series to get the
            actual shift, rounded up. A callable must take an integer,
            the time series length, as input and return a corresponding
            integer shift. Defaults to 1.
        depth (int, optional): The number of times this transform is
            applied, e.g. ``depth=2`` corresponds to the second order
            increments. Defaults to 1.
        zero_padding (bool, optional): If set to True, then the first
            entry in each time series will be set to 0. If False, it is
            set to the first value of the original time series. Defaults
            to True.
    """

    def __init__(
        self,
        shift: Union[int, float, Callable[[int], int]] = 1,
        depth: int = 1,
        zero_padding: bool = True,
    ) -> None:
        self._shift = shift
        if depth < 1:
            raise ValueError("depth has to be a positive integer > 0")
        self._depth = depth
        self._zero_padding = zero_padding

    @property
    def requires_fitting(self) -> bool:
        return False

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self._shift, int):
            shift = self._shift
        elif isinstance(self._shift, float):
            shift = np.ceil(self._shift * X.shape[2])
        elif callable(self._shift):
            shift = self._shift(X.shape[2])
        else:
            raise TypeError(f"Type {type(self._shift)} "
                            f"not supported for argument shift")
        out = X
        for _ in range(self._depth):
            out = _increments(out, shift)
            if not self._zero_padding:
                out[:, :, :self._shift] = X[:, :, :self._shift]
        return out

    def _copy(self) -> "INC":
        return INC(self._shift, self._depth, self._zero_padding)

    def __eq__(self, other) -> bool:
        if (isinstance(other, INC)
                and self._shift == other._shift
                and self._depth == other._depth
                and self._zero_padding == other._zero_padding):
            return True
        return False

    def __str__(self) -> str:
        return f"INC({self._shift}, {self._depth}, {self._zero_padding})"


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
        std_eps (float, optional): Error to add to the standard
            deviation before dividing the (centered) time series by it.
            This avoids 'division by zero' errors and will reduce the
            number of large values after this transform, which otherwise
            might explode in further calculations of iterated sums.
            Defaults to ``1e-5``.
    """

    def __init__(
        self,
        separately: bool = True,
        var: bool = True,
        std_eps: float = 1e-5,
    ) -> None:
        self._separately = separately
        self._div_std = var
        self._mean = None
        self._std = None
        self._eps = std_eps

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
            out = (X - self._mean) / (self._std + self._eps)
        else:
            mean_ = np.mean(X, axis=2)[:, :, np.newaxis]
            std_ = np.ones((X.shape[0], X.shape[1], 1))
            if self._div_std:
                std_ = np.std(X, axis=2)[:, :, np.newaxis]
            out = (X - mean_) / (std_ + self._eps)
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
        kernel (np.ndarray, optional): If set to a numpy array, the
            kernel is taken as given. This will ignore all other
            arguments of the class except ``adaptive_width``.
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
        kernel: Optional[np.ndarray] = None,
    ) -> None:
        self._width = width
        self._adaptive_width = adaptive_width
        self._out_dim = out_dim
        self._force_sum_one = force_sum_one
        self._const_kernel = kernel

    def _fit(self, X: np.ndarray) -> None:
        if self._const_kernel is not None:
            self._kernel = self._const_kernel.copy()
            self._ndim_per_kernel = np.ones((X.shape[1],), dtype=np.int32)
            self._dims_per_kernel = np.arange(X.shape[1], dtype=np.int32)
            return

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
            kernel=self._const_kernel,
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RIN) and (self._width == other._width
                and self._adaptive_width == other._adaptive_width
                and self._out_dim == other._out_dim
                and self._force_sum_one == other._force_sum_one
                and self._const_kernel == other._const_kernel):
            return True
        return False

    def __str__(self) -> str:
        return (
            f"RIN({self._width}, {self._adaptive_width}, "
            f"{self._out_dim}, {self._force_sum_one}, {self._const_kernel})"
        )


class RDW(Preparateur):
    """Preparateur: Random Dimension Weights

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
    """Preparateur: Johnson-Lindenstrauss Dimensionality Reduction

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


class SPE(Preparateur):
    """Preparateur: Sinusoidal Positional Embedding

    Transforms a time series by multiplying it with a sine wave function
    of a certain frequency.::

        y_t = x_t * sin(t / T**f), t = 1, ..., T

    Args:
        freq (float): Frequency parameter ``f`` of the sine wave. This
            should be a value between 0 and 1.
        operation (str, optional): Type of the operation used for the
            embedding. Has to be either 'additive' or 'multiplicative'.
            Defaults to 'multiplicative'.
        function (callable, optional): A callable that is used to
            transform time steps. The results of this callable are then
            added or multiplied to the input time series. Defaults to a
            sine function.
        step_transform (str, optional): If set to 'L1' or 'L2', the
            cumulative sum of the normed increments is used in the sine
            function instead of ``t``.
        max_length (int, optional): The parameter ``T`` in the
            denominator controlling the strength of frequency changes of
            the sine function. Defaults to the time series length.
    """

    def __init__(
        self,
        freq: float,
        operation: Literal["additive", "multiplicative"] = "multiplicative",
        function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        step_transform: Optional[Literal["L1", "L2"]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self._freq = freq
        self._operation: Literal["additive", "multiplicative"] = operation
        self._function = function
        self._step_transform: Optional[Literal["L1", "L2"]] = step_transform
        self._max_length = max_length

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._step_transform is None:
            T = X.shape[2] if self._max_length is None else self._max_length
            range_ = np.arange(X.shape[2]) / (T**self._freq)
        else:
            range_ = self._cache.get(CacheType.ISS, self._step_transform, X)
            T = range_[:, -1:] if self._max_length is None else (
                self._max_length
            )
            range_ = (range_ / (T**self._freq))
        if self._function is None:
            wave = np.sin(range_)
        else:
            wave = self._function(range_)
        wave = (
            wave[np.newaxis, np.newaxis, :]
            if self._step_transform is None else wave[:, np.newaxis, :]
        )
        if self._operation == "multiplicative":
            return X * wave
        elif self._operation == "additive":
            return X + wave
        else:
            raise ValueError(f"Unknown operation given: {self._operation}")

    def _copy(self) -> "SPE":
        return SPE(
            freq=self._freq,
            operation=self._operation,
            function=self._function,
            step_transform=self._step_transform,
            max_length=self._max_length,
        )

    def __eq__(self, other: Any) -> bool:
        if (isinstance(other, SPE)
                and self._freq == other._freq
                and self._operation == other._operation
                and self._function == other._function
                and self._step_transform == other._step_transform
                and self._max_length == other._max_length):
            return True
        return False

    def __str__(self) -> str:
        return (f"SPE({self._freq}, {self._operation}, {self._function}, "
                f"{self._step_transform}, {self._max_length})")


class RPE(Preparateur):
    """Preparateur: Rotational Positional Embedding

    Transforms a two dimensional time series by multiplying it with a
    rotation matrix ``R_t``.

    Args:
        freq (float): Frequency parameter ``f`` of the rotation. This
            should be a value between 0 and 1.
        step_transform (str, optional): If set to 'L1' or 'L2', the
            cumulative sum of the normed increments is used in the sine
            function instead of ``t``.
        max_length (int, optional): The parameter ``T`` in the
            denominator controlling the strength of frequency changes of
            the sine function. Defaults to the time series length.
    """

    @staticmethod
    @numba.njit(
        "f8[:,:,:](f8[:,:,:], i4, f8)",
        fastmath=True,
        cache=True,
        parallel=True,
    )
    def _backend(
        X: np.ndarray,
        T: int,
        freq: float,
    ) -> np.ndarray:
        result = np.zeros((X.shape[0], 2, X.shape[2]))
        for i in numba.prange(X.shape[0]):
            for k in numba.prange(X.shape[2]):
                result[i, 0, k] = (
                    np.cos(k / T**freq) * X[i, 0, k] -
                    np.sin(k / T**freq) * X[i, 1, k]
                )
                result[i, 1, k] = (
                    np.sin(k / T**freq) * X[i, 0, k] +
                    np.cos(k / T**freq) * X[i, 1, k]
                )
        return result

    def __init__(
        self,
        freq: float,
        max_length: Optional[int] = None,
    ) -> None:
        self._freq = freq
        self._max_length = max_length

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != 2:
            raise ValueError(
                f"RPE input has to have 2 dimensions, got {X.shape[1]}"
            )
        return RPE._backend(
            X,
            X.shape[2] if self._max_length is None else self._max_length,
            self._freq,
        )

    def _copy(self) -> "RPE":
        return RPE(freq=self._freq, max_length=self._max_length)

    def __eq__(self, other: Any) -> bool:
        if (isinstance(other, RPE)
                and self._freq == other._freq
                and self._max_length == other._max_length):
            return True
        return False

    def __str__(self) -> str:
        return f"RPE({self._freq}, {self._max_length})"


class FUN(Preparateur):
    """Preparateur: Function Transform

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
