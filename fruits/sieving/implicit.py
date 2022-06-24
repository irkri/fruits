from typing import Union

import numpy as np

from fruits.sieving.abstract import FeatureSieve
from fruits._backend import _increments


class PPV(FeatureSieve):
    """FeatureSieve: Proportion of positive values

    For a calculated quantile with ``PPV.fit``, this feature sieve
    calculates the proportion of values in a time series that are
    greater than the calculated quantile(s).

    :param quantile: Quantile or list of quantiles ``[q_1, ..., q_n]``
        as actual value(s) or probability for quantile calculation
        (e.g. 0.5 for the 0.5-quantile)., defaults to 0.5
    :type quantile: float or list of floats, optional
    :param constant: If ``True``, the argument ``quantile`` is
        interpreted as the actual value for the quantile.
        If ``quantile`` is a list, then ``constant`` can be a list of
        booleans ``[b_1, ..., b_n]`` where ``b_i`` explains the
        interpretation of ``q_i`` or a single boolean for each ``q_i``.
        (value or probability)., defaults to False
    :type constant: bool or list of bools, optional
    :param sample_size: Sample size to use for quantile calculation.
        This option can be ignored if ``constant`` is set to ``True``.,
        defaults to 1.0
    :type sample_size: float, optional
    :param segments: If `True`, then the proportion of values within
        each two quantiles will be calculated. If `quantile` is a list,
        then this list will be sorted and the corresponding features
        will be

        .. code-block::python
            np.array([np.sum(q_{i-1} <= X[k] < q_i)]) / len(X[k])])

        where ``k`` is the index of the time series and ``i`` ranges
        from 1 to n.
        If set to ``False``, then the features will be

        .. code-block::python
            np.array([np.sum(X[k] <= q_i)]) / len(X[k])])

        with the same index rules., defaults to False
    :type segments: bool, optional
    """

    def __init__(
        self,
        quantile: Union[list[float], float] = 0.5,
        constant: Union[list[bool], bool] = False,
        sample_size: float = 1.0,
        segments: bool = False,
    ):
        super().__init__("Proportion of positive values")
        if isinstance(quantile, list):
            if not isinstance(constant, list):
                constant = [constant for _ in range(len(quantile))]
            elif len(quantile) != len(constant):
                raise ValueError("If 'quantile' is a list, then 'constant' "
                                 + "also has to be a list of same length or "
                                 + "a single boolean.")
            for q, c in zip(quantile, constant):
                if not c and not 0 <= q <= 1:
                    raise ValueError("If 'constant' is set to False, "
                                     + "'quantile' has to be a value in [0,1]")
        else:
            quantile = [quantile]
            if isinstance(constant, list):
                if len(constant) > 1:
                    raise ValueError("'constant' has to be a single boolean"
                                     + "if 'quantile' is a single float")
            else:
                constant = [constant]
        if segments:
            self._q_c_input = list(zip(list(set(quantile)), constant))
            self._q_c_input = sorted(self._q_c_input, key=lambda x: x[0])
        else:
            self._q_c_input = list(zip(quantile, constant))
        self._q: list[float]
        if not 0 < sample_size <= 1:
            raise ValueError("'sample_size' has to be a float in (0, 1]")
        self._sample_size = sample_size
        if segments and len(quantile) == 1:
            raise ValueError("If 'segments' is set to `True` then 'quantile'"
                             + "has to be a list of length >= 2.")
        self._segments = segments

    def nfeatures(self) -> int:
        """Returns the number of features this sieve produces.

        :rtype: int
        """
        if self._segments:
            return len(self._q_c_input) - 1
        else:
            return len(self._q_c_input)

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Calculates and remembers the quantile(s) of the input data.

        :type X: np.ndarray
        """
        self._q = [x[0] for x in self._q_c_input]
        for i, q in enumerate(self._q):
            if not self._q_c_input[i][1]:
                sample_size = int(self._sample_size * len(X))
                if sample_size < 1:
                    sample_size = 1
                selection = np.random.choice(
                    np.arange(len(X)),
                    size=sample_size,
                    replace=False,
                )
                self._q[i] = np.quantile(
                    np.array([X[i] for i in selection]).flatten(),
                    q,
                )

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :type X: np.ndarray
        :returns: array of features
        :rtype: np.ndarray
        :raises: RuntimeError if ``self.fit`` wasn't called
        """
        if not hasattr(self, "_q"):
            raise RuntimeError("Missing call of PPV.fit()")
        result = np.zeros((X.shape[0], self.nfeatures()))
        if self._segments:
            for j in range(1, len(self._q)):
                result[:, j-1] = np.sum(
                    np.logical_and(self._q[j-1] <= X, X < self._q[j]),
                    axis=1
                )
                result[:, j-1] /= X.shape[1]
        else:
            for j, q in enumerate(self._q):
                result[:, j] = np.sum((X >= q), axis=1)
                result[:, j] /= X.shape[1]
        return result

    def copy(self) -> "PPV":
        """Returns a copy of this object.

        :rtype: PPV
        """
        fs = PPV([x[0] for x in self._q_c_input],
                 [x[1] for x in self._q_c_input],
                 self._sample_size,
                 self._segments)
        return fs

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        string = f"PPV [sampling={self._sample_size}"
        if self._segments:
            string += ", segments"
        string += f"] -> {self.nfeatures()}:"
        for x in self._q_c_input:
            string += f"\n   > {x[0]} | {x[1]}"
        return string

    def __str__(self) -> str:
        return "PPV(" + \
               f"quantile={[x[0] for x in self._q_c_input]}, " + \
               f"constant={[x[1] for x in self._q_c_input]}, " + \
               f"sample_size={self._sample_size}, " + \
               f"segments={self._segments})"


class CPV(PPV):
    """FeatureSieve: Proportion of connected components of positive
    values

    For a calculated quantile with ``CPV.fit``, this FeatureSieve
    calculates the connected components of the proportion of values in
    a time series that is greater than the calculated quantile.
    This is equivalent to the number of consecutive strips of 1's in
    the array ``X>=quantile``.

    :param quantile: Quantile or list of quantiles ``[q_1, ..., q_n]``
        as actual value(s) or probability for quantile calculation
        (e.g. 0.5 for the 0.5-quantile)., defaults to 0.5
    :type quantile: float or list of floats, optional
    :param constant: If ``True``, the argument ``quantile`` is
        interpreted as the actual value for the quantile.
        If ``quantile`` is a list, then ``constant`` can be a list of
        booleans ``[b_1, ..., b_n]`` where ``b_i`` explains the
        interpretation of ``q_i`` or a single boolean for each ``q_i``.
        (value or probability)., defaults to False
    :type constant: bool or list of bools, optional
    :param sample_size: Sample size to use for quantile calculation.
        This option can be ignored if ``constant`` is set to ``True``.,
        defaults to 0.05
    :type sample_size: float, optional
    """

    def __init__(self,
                 quantile: Union[list[float], float] = 0.5,
                 constant: Union[list[bool], bool] = False,
                 sample_size: float = 1.0,
                 segments: bool = False):
        super().__init__(quantile,
                         constant,
                         sample_size,
                         segments)
        self.name = "Proportion of connected components of positive values"

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed data. See the class definition for
        detailed information.

        :type X: np.ndarray
        :returns: Array of features.
        :rtype: np.ndarray
        :raises: RuntimeError if ``self.fit`` wasn't called
        """
        if not hasattr(self, "_q"):
            raise RuntimeError("Missing call of CPV.fit()")
        result = np.zeros((X.shape[0], self.nfeatures()))
        n = X.shape[1]
        if n % 2 == 1:
            n += 1
        if self._segments:
            for j in range(1, len(self._q)):
                diff = _increments(np.expand_dims(
                                    np.logical_and(
                                            self._q[j-1] <= X,
                                            X < self._q[j]).astype(np.float64),
                                    axis=1))[:, 0, :]
                result[:, j-1] = 2 * np.sum(diff == 1, axis=-1) / n
        else:
            for j, q in enumerate(self._q):
                diff = _increments(
                    np.expand_dims((X >= q).astype(np.float64), axis=1)
                )[:, 0, :]
                result[:, j] = 2 * np.sum(diff == 1, axis=-1) / n
        return result

    def copy(self) -> "CPV":
        """Returns a copy of this object.

        :rtype: CPV
        """
        fs = CPV([x[0] for x in self._q_c_input],
                 [x[1] for x in self._q_c_input],
                 self._sample_size,
                 self._segments)
        return fs

    def summary(self) -> str:
        """Returns a better formatted summary string for the sieve."""
        string = f"CPV [sampling={self._sample_size}"
        if self._segments:
            string += ", segments"
        string += f"] -> {self.nfeatures()}:"
        for x in self._q_c_input:
            string += f"\n   > {x[0]} | {x[1]}"
        return string

    def __str__(self) -> str:
        string = "CPV(" + \
                f"quantile={[x[0] for x in self._q_c_input]}, " + \
                f"constant={[x[1] for x in self._q_c_input]}, " + \
                f"sample_size={self._sample_size}, " + \
                f"segments={self._segments})"
        return string
