import numpy as np

from fruits.base.callback import AbstractCallback

def check_input_shape(X: np.ndarray) -> bool:
    """Checks if the given time series dataset has the correct input
    shape.
    
    :type X: np.ndarray
    :rtype: bool
    """
    if X.ndim == 3:
        return True
    return False

def force_input_shape(X: np.ndarray):
    """Makes the attempt to format the input shape of the
    multidimensional time series dataset ``X``.
    This leads to an three dimensional array where

    - ``X.shape[0]``: Number of time series
    - ``X.shape[1]``: Number of dimensions in each time series
    - ``X.shape[2]``: Length of each time series
    
    :type X: np.ndarray
    :rtype: np.ndarray
    :raises: ValueError if ``X.ndim > 3``
    """
    out = X.copy()
    if out.ndim == 1:
        out = np.expand_dims(out, axis=0)
    if out.ndim == 2:
        out = np.expand_dims(out, axis=1)
    if out.ndim != 3:
        raise ValueError("Unsupported input shape")
    return out

def check_callbacks(callbacks: list):
    """For a given list of objects, checks if all objects are inheriting
    from :class:`~fruits.base.callback.AbstractCallback`. If not, a
    ``TypeError`` will be raised.
    
    :type callbacks: list
    """
    for callback in callbacks:
        if not isinstance(callback, AbstractCallback):
            raise TypeError("Supplied callback is not of type " +
                            "AbstractCallback")
