import warnings
from typing import Protocol

import numpy as np


class FitTransform(Protocol):
    """Protocol for a class that supports fitting and transforming a
    time series dataset.
    """

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the object to the given time series dataset.

        :type X: np.ndarray
        """

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transforms the given time series dataset.

        :type X: np.ndarray
        :returns: Transformed time series dataset.
        :rtype: np.ndarray
        """


def force_input_shape(X: np.ndarray) -> np.ndarray:
    """Makes the attempt to format the input shape of the given
    multidimensional time series dataset.
    This leads to an three dimensional array where

    - ``X.shape[0]``: Number of time series
    - ``X.shape[1]``: Number of dimensions in each time series
    - ``X.shape[2]``: Length of each time series

    :type X: np.ndarray
    :rtype: np.ndarray
    :raises: ValueError if ``X.ndim > 3``
    """
    if X.ndim < 3:
        warnings.warn("The input shape of the time series dataset "
                      "is < 3 and needs to be converted.")
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        return X
    elif X.ndim == 3:
        return X.copy()
    else:
        raise ValueError("Unsupported input shape of the given time series "
                         "dataset")
