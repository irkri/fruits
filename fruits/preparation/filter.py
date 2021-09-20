from typing import Union

import numpy as np

from fruits.preparation.abstract import DataPreparateur

class DIL(DataPreparateur):
    """DataPreparateur: Dilation

    This preparateur sets some slices in each time series in the
    given dataset to zero. The indices and lengths for those zero
    sequences are chosen randomly.

    :param clusters: If a float value between 0 and 1 (incl.) is given,
        the number of zero strips will be calculated by multiplying
        ``clusters * X.shape[2]`` in ``self.fit(X)``.
        If ``None``, this number will be a random integer between ``1``
        and ``numpy.floor(X.shape[2] / 10.0) - 1`` instead.,
        defaults to None
    :type clusters: Union[float, None], optional
    """
    def __init__(self, clusters: Union[float, None] = None):
        super().__init__("Dilation")
        self._clusters = clusters
        self._indices = None
        self._lengths = None
    
    def fit(self, X: np.ndarray):
        """Fits the preparateur to the given dataset by randomizing the
        starting points and lengths of the zero strips.

        :type X: np.ndarray
        """
        if self._clusters is not None:
            nclusters = int(self._clusters * X.shape[2])
        else:
            nclusters = np.random.randint(1, int(np.floor(X.shape[2] / 10.0)))
        if nclusters >= X.shape[2]:
            self._indices = np.arange(X.shape[2])
        else:
            self._indices = np.sort(np.random.choice(X.shape[2],
                                                     size=nclusters,
                                                     replace=False))
        self._lengths = []
        for i in range(nclusters):
            if i == nclusters - 1:
                max_length = X.shape[2] - self._indices[i]
            else:
                max_length = self._indices[i+1] - self._indices[i]
            self._lengths.append(np.random.randint(1, max_length+1))
            
    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if self._indices is None or self._lengths is None:
            raise RuntimeError("Missing call of self.fit()")
        X_new = X.copy()
        for i in range(len(self._indices)):
            X_new[:, :, self._indices[i]:self._indices[i]+self._lengths[i]] = 0
        return X_new
    
    def copy(self) -> "DIL":
        """Returns a copy of this preparateur.

        :rtype: DIL
        """
        return DIL(self._clusters)

    def __eq__(self, other) -> bool:
        return False

    def __str__(self) -> str:
        return f"DIL(clusters={self._clusters})"

    def __repr__(self) -> str:
        return "fruits.preparation.filter.DIL"


class WIN(DataPreparateur):
    """DataPreparateur: Window

    Outside of a certain window the time series is set to zero.
    The window is obtained according to 'quantiles' of a certain
    function of each time series, for example its quadratic variation by
    calculating increments on the results from
    ``fruits.core.ISS(X, [SimpleWord("[11]")])``.

    :param start: Quantile start; float value between 0 and 1 (incl.).
    :type start: float
    :param end: Quantile end; float value between 0 and 1 (incl.).
    :type end: float
    """
    def __init__(self,
                 start: float,
                 end: float):
        super().__init__("Window")
        self._start = start
        self._end = end
        self._requisite = "INC -> [11]"

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        Q = self._get_requisite(X)

        maxima = np.expand_dims(np.max(Q, axis=2), axis=-1)
        Q = Q / maxima

        mask = (Q >= self._start) & (Q <= self._end)
        return X * mask
    
    def copy(self) -> "WIN":
        """Returns a copy of this preparateur.

        :rtype: WIN
        """
        return WIN(self._start, self._end)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WIN):
            raise TypeError(f"Cannot compare WIN with type {type(other)}")
        return (self._start == other._start and
                self._end == other._end)

    def __str__(self) -> str:
        return f"WIN(start={self._start}, end={self._end})"

    def __repr__(self) -> str:
        return "fruits.preparation.filter.WIN"


class DOT(DataPreparateur):
    """DataPreparateur: Dotting

    Keeps every ``n``-th point of a time series while setting everything
    else to zero.

    :param n: If an integer is given, this value will be directly used
        for the described purpose. If a float between 0 and 1 is given,
        the actual value for ``n`` will be calculated in ``self.fit``
        by ``n=n_prop*X.shape[2]``., defaults to 2
    :type n: Union[int, float], optional
    :param first: First index to keep. If set to ``None``, the value
        will be internally set to ``n``. Can be either a float or an
        integer with the same rules as argument ``n``., defaults to None
    :type first: Union[int, float, None], optional
    """
    def __init__(self,
                 n: Union[int, float] = 2,
                 first: Union[int, float, None] = None):
        super().__init__("Dotting")
        if isinstance(n, float) and not 0 < n < 1:
            raise ValueError("If n is a float, it has to satisfy 0 < n < 1")
        elif not isinstance(n, float) and not isinstance(n, int):
            raise TypeError("n has to be either a float or integer")

        self._n_given = n
        self._n = None

        if isinstance(first, float) and not 0 < first < 1:
            raise ValueError("If first is a float, "
                             + " it has to satisfy 0 < first < 1")
        elif (not isinstance(first, float)
              and not isinstance(first, int)
              and first is not None):
            raise TypeError("first has to be either a float, integer or None")

        self._first_given = first
        self._first = None

    def fit(self, X: np.ndarray):
        """Fits the preparateur to the given dataset by (if necessary)
        calculating the value of ``n``.

        :type X: np.ndarray
        """
        if isinstance(self._n_given, float):
            self._n = int(self._n_given * X.shape[2])
            if self._n <= 0:
                self._n = 1
        else:
            if self._n_given >= X.shape[2]:
                self._n = X.shape[2]
            else:
                self._n = self._n_given
        if isinstance(self._first_given, float):
            self._first = int(self._first_given * X.shape[2])
            if self._first <= 0:
                self._first = 1
            if self._first >= X.shape[2]:
                self._first = X.shape[2] - 1
        elif self._first_given is not None:
            if self._first_given >= X.shape[2]:
                self._first = X.shape[2] - 1
            else:
                self._first = self._first_given
        else:
            self._first = self._n - 1

    def prepare(self, X: np.ndarray):
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        if self._n is None:
            raise RuntimeError("Missing call of self.fit()")
        out = np.zeros(X.shape)
        out[:, :, self._first::self._n] = X[:, :, self._first::self._n]
        return out
    
    def copy(self) -> "DOT":
        """Returns a copy of this preparateur.

        :rtype: DOT
        """
        return DOT(self._n_given, self._first_given)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DOT):
            raise TypeError(f"Cannot compare DOT with type {type(other)}")
        return ((self._n_given == other._n_given)
                and (self._first_given == other._first_given))

    def __str__(self) -> str:
        return f"DOT(n={self._n_given}, first={self._first_given})"

    def __repr__(self) -> str:
        return "fruits.preparation.filter.DOT"
