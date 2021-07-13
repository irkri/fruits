"""This small python module is an appendix to the package FRUITS and
allows more ways for constructing ComplexWord objects.
"""

import numpy as np

from context import fruits

# complex letters implementation

def get_dilated_letter(nclusters: int = 100):
    _indices = sorted(np.random.random_sample(nclusters))
    _lengths = []
    for i in range(len(_indices)):
        if i == len(_indices)-1:
            b = 1 - _indices[i]
        else:
            b = _indices[i+1] - _indices[i]
        _lengths.append(b*np.random.random_sample())

    @fruits.core.complex_letter(name="DILATED")
    def dilated(X: np.ndarray, i: int):
        X_ = X.copy()
        for i in range(min(int(0.01*X.shape[2]), len(_indices))):
            start = int(_indices[i] * X.shape[2])
            length = int(_lengths[i] * X.shape[2])
            X_[:, :, start:start+length] = 0
        return X_

    return dilated

@fruits.core.complex_letter(name="SIGMOID")
def sigmoid(X: np.ndarray, i: int):
    return 1 / (1 + np.exp(-0.001*X[i, :]))

@fruits.core.complex_letter(name="leakyRELU")
def leaky_relu(X: np.ndarray, i: int):
    out = np.zeros(X.shape[1], dtype=np.float64)
    out += X[i, :] * (X[i, :]>0)
    out += (X[i, :]*0.005) * (X[i, :]<=0)
    return out

@fruits.core.complex_letter(name="TANH")
def tanh(X: np.ndarray, i: int):
    pos = np.exp(0.001*X[i, :])
    neg = np.exp(-0.001*X[i, :])
    return (pos-neg) / (pos+neg)

@fruits.core.complex_letter(name="ID")
def id_(X: np.ndarray, i: int):
    return X[i, :]
