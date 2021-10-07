from typing import Union, List

import numpy as np

from fruits.preparation.abstract import DataPreparateur
from fruits._backend import _coquantile


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
        self._indices: np.ndarray
        self._lengths: List[int]

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the preparateur to the given dataset by randomizing the
        starting points and lengths of the zero strips.

        :type X: np.ndarray
        """
        if self._clusters is not None:
            nclusters = int(self._clusters * X.shape[2])
        else:
            upper_bound = int(np.floor(X.shape[2] / 10.0))
            if upper_bound <= 1:
                nclusters = 1
            else:
                nclusters = np.random.randint(1, upper_bound)
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

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        :raises: RuntimeError if self.fit() wasn't called
        """
        if not hasattr(self, "_indices") or not hasattr(self, "_lengths"):
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

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        coq_start = _coquantile(X.astype(np.float64), self._start)
        coq_end = _coquantile(X.astype(np.float64), self._end)
        print(coq_start, coq_end)
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j, coq_start[i]-1:coq_end[i]] = (
                    X[i, j, coq_start[i]-1:coq_end[i]]
                )
        return result

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
        self._n: int

        if isinstance(first, float) and not 0 < first < 1:
            raise ValueError("If first is a float, "
                             + " it has to satisfy 0 < first < 1")
        elif (not isinstance(first, float)
              and not isinstance(first, int)
              and first is not None):
            raise TypeError("first has to be either a float, integer or None")

        self._first_given = first
        self._first: int

    def fit(self, X: np.ndarray, **kwargs) -> None:
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

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        if not hasattr(self, "_n") or not hasattr(self, "_first"):
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


class PDD(DataPreparateur):
    """DataPreparateur: Proportion-Density-Drop

    Sets values in the given time series to zero. The number of values
    and their distribution in the time domain is adjustable.

    :param density: A float in ``(0,1]``. A low value translates to a
        larger gap between the points that are being set to zero.,
        defaults to 0.1
    :type density: float, optional
    :param proportion: Proportion of the length of each time series to
        drop. Has to be a float in ``(0,1)``., defaults to 0.5
    :type proportion: float, optional
    """

    def __init__(self,
                 density: float = 0.1,
                 proportion: float = 0.5):
        super().__init__("Proportion-Density-Drop")
        if not isinstance(density, float) or not 0.0 < density <= 1.0:
            raise ValueError("density has to be a float 0 < density <= 1")
        if not isinstance(proportion, float) or not 0.0 < proportion < 1.0:
            raise ValueError("proportion has to be a float 0 < proportion < 1")

        self._d_given = density
        self._p_given = proportion
        self._indices: np.ndarray
        self._width: int

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fits the preparateur to the given dataset by calculating the
        actual values of ``density`` and ``proportion``.

        :type X: np.ndarray
        """
        p = int(self._p_given * X.shape[2])
        if p < 1:
            p = 1
        points = int((1.0 - self._d_given) * X.shape[2])
        if points < 1:
            points = 1
        self._width = int(p / points)
        if points == X.shape[2]-self._width:
            points -= 1
        self._indices = np.linspace(0, X.shape[2]-self._width, points,
                                    dtype="int")

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the transformed dataset.

        :type X: np.ndarray
        :rtype: np.ndarray
        """
        if not hasattr(self, "_width") or not hasattr(self, "_indices"):
            raise RuntimeError("Missing call of self.fit()")
        out = X.copy()
        for i in range(len(self._indices)):
            out[:, :, self._indices[i]:self._indices[i]+self._width] = 0
        return out

    def copy(self) -> "PDD":
        """Returns a copy of this preparateur.

        :rtype: PDD
        """
        return PDD(self._d_given, self._p_given)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PDD):
            raise TypeError(f"Cannot compare PDD with type {type(other)}")
        return ((self._d_given == other._d_given)
                and (self._p_given == other._p_given))

    def __str__(self) -> str:
        return f"PDD(density={self._d_given}, proportion={self._p_given})"

    def __repr__(self) -> str:
        return "fruits.preparation.filter.PDD"
