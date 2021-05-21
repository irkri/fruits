import numpy as np
from copy import copy
import numba

class FeatureSieve:
    """Class FeatureSieve
    
    A FeatureSieve object is used to extract a single number out of an
    multidimensional numpy array.
    """
    def __init__(self, name:str=""):
        self._name = name
        self._args = ()
        self._kwargs = {}
        self._func = None

    def set_function(self, f):
        if not callable(f):
            raise TypeError("Cannot set non-callable object as function")
        self._func = f

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name:str):
        self._name = name

    def __repr__(self) -> str:
        out = "FeatureSieve('"+self._name+"'"
        if self._args:
            out += ","+",".join([str(x) for x in self._args])
        if self._kwargs:
            out += ","+",".join([str(x)+"="+str(self._kwargs[x]) 
                                 for x in self._kwargs])
        out += ")"
        return out
    
    def __call__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        return copy(self)

    def __copy__(self):
        ff = FeatureSieve(self.name)
        ff._args = self._args
        ff._kwargs = self._kwargs
        self._args = ()
        self._kwargs = {}
        ff.set_function(self._func)
        return ff
        
    def sieve(self, X:np.ndarray):
        if self._func is None:
            raise RuntimeError("No function specified")
        X = np.atleast_2d(X)
        return self._func(X, *self._args, **self._kwargs)

@numba.njit(parallel=True, fastmath=True)
def _fast_ppv(X:np.ndarray, ref_value:float) -> np.ndarray:
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        c = 0
        for j in range(len(X[i])):
            if X[i][j]>=ref_value:
                c += 1
        if len(X[i])==0:
            result[i] = 0
        else:
            result[i] = c/len(X[i])
    return result

def _ppv(X:np.ndarray,
         quantile:float=0.5,
         constant:bool=False,
         sample_size:float=0.05) -> np.ndarray:
    if constant:
        ref_value = quantile
    else:
        if not 0<quantile<1:
            raise ValueError("If 'constant' is set to False, quantile has "+
                             "to be a value between 0 and 1")
        sample_size = int(sample_size*len(X))
        sample_size = 1 if sample_size<1 else sample_size
        selection = np.random.choice(np.arange(len(X)), size=sample_size,
                                     replace=False)
        ref_value = np.quantile(np.array([X[i] for i in selection]).flatten(),
                                quantile)
    if len(X)==0:
        return 0
    return _fast_ppv(X, ref_value)

PPV = FeatureSieve("proportion of positive values")
PPV.set_function(_ppv)

def _ppv_connected_components( X:np.ndarray,
                             quantile:float=0.5,
                             constant:bool=False,
                             sample_size:float=0.05) -> np.ndarray:
    """Count connected components, i.e. the number of consecutive strips of 1's."""
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    if len(X)==0:
        return 0
    if constant:
        ref_value = quantile
    else:
        if not 0<quantile<1:
            raise ValueError("If 'constant' is set to False, quantile has "+
                             "to be a value between 0 and 1")
        sample_size = int(sample_size*len(X))
        sample_size = 1 if sample_size<1 else sample_size
        selection = np.random.choice(np.arange(len(X)), size=sample_size,
                                     replace=False)
        ref_value = np.quantile(np.array([X[i] for i in selection]).flatten(),
                                quantile)

    positive = np.pad( (X > ref_value).astype(int), ( (0,0), (1,0) ), 'constant', constant_values=0 )
    diff = positive[:,1:] - positive[:,:-1]
    s = np.sum( diff == 1, axis=-1)
    return 2 * s / X.shape[1] # At most X.shape[1]/2 connected components are possible.

PPV_connected = FeatureSieve("proportion of connected components of positive values")
PPV_connected.set_function(_ppv_connected_components)

def get_ppv(n:int=1, a:float=0, b:float=1, 
            constant:bool=False, sample_size:float=0.05) -> list:
    """Returns a list of PPV feature sieves.
    
    :param n: Number of sieves (quantiles will be evenly spaced numbers
    between `a` and `b`), defaults to 1
    :type n: int, optional
    :param a: Left interval border where the quantiles will be drawn
    from, defaults to 0
    :type a: float, optional
    :param b: Left interval border where the quantiles will be drawn
    from, defaults to 1
    :type b: float, optional
    :param constant: if `True`, the quantiles will be interpreted as
    actual numbers, not percentiles, defaults to False
    :type constant: bool, optional
    :param sample_size: The quantiles will be calculated on this 
    proportion of the input data, Only matters if `constant` is set to
    `False`, defaults to 0.05
    :type sample_size: float, optional
    """
    return [PPV(q, constant=constant, sample_size=sample_size)
            for q in np.linspace(a, b, num=n)]

@numba.njit(parallel=True, fastmath=True)
def _max(X:np.ndarray):
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        if len(X[i])==0:
            continue
        maximum = X[i][0]
        for j in range(len(X[i])):
            if X[i][j]>maximum:
                maximum = X[i][j]
        result[i] = maximum
    return result

MAX = FeatureSieve("maximal value")
MAX.set_function(_max)

@numba.njit(parallel=True, fastmath=True)
def _min(X:np.ndarray):
    result = np.zeros(len(X))
    for i in numba.prange(len(X)):
        if len(X[i])==0:
            continue
        minimum = X[i][0]
        for j in range(len(X[i])):
            if X[i][j]<minimum:
                minimum = X[i][j]
        result[i] = minimum
    return result

MIN = FeatureSieve("minimal value")
MIN.set_function(_min)

def _end(X:np.ndarray):
    return X[:, -1]

END = FeatureSieve("last value")
END.set_function(_end)
