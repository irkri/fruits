import numpy as np

from fruits.base.callback import AbstractCallback

def force_input_shape(X: np.ndarray):
    out = X.copy()
    if out.ndim == 1:
        out = np.expand_dims(out, axis=0)
    if out.ndim == 2:
        out = np.expand_dims(out, axis=1)
    if out.ndim != 3:
        raise ValueError("Unsupported input shape")
    return out

def check_callbacks(callbacks: list):
    for callback in callbacks:
        if not isinstance(callback, AbstractCallback):
            raise TypeError("Supplied callback is not of type " +
                            "AbstractCallback")
