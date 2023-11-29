from itertools import product
from typing import Generator, Optional, Sequence

import numba
import numpy as np

from .iss import ISS
from .words.word import SimpleWord, Word


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f4, i4[:,:])",
    fastmath=True,
    cache=True,
)
def _coswiss_single(
    X: np.ndarray,
    word: np.ndarray,
    freq: float,
    weightings: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[1], ), dtype=np.float64)
    sin_w = np.sin(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    cos_w = np.cos(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    for i in range(weightings.shape[0]):
        tmp = np.ones((X.shape[1], ), dtype=np.float64)
        for k, extended_letter in enumerate(word):
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
            for letter, occurence in enumerate(extended_letter):
                if occurence > 0:
                    for _ in range(occurence):
                        tmp = tmp * X[letter, :]
                elif occurence < 0:
                    for _ in range(-occurence):
                        tmp = tmp / X[letter, :]
            for _ in range(weightings[i, 2*k+1]):
                tmp = tmp * sin_w
            for _ in range(weightings[i, 2*k+2]):
                tmp = tmp * cos_w
            tmp = np.cumsum(tmp)
        if weightings.shape[1] == 2*len(word) + 3:
            for _ in range(weightings[i, 2*len(word)+1]):
                tmp = tmp * sin_w
            for _ in range(weightings[i, 2*len(word)+2]):
                tmp = tmp * cos_w
        result += weightings[i, 0] * tmp
    return result


@numba.njit(
    "f8[:](f8[:,:], i4[:,:], f4, i4[:,:], i4[:,:])",
    fastmath=True,
    cache=True,
)
def _leaky_coswiss_single(
    X: np.ndarray,
    word: np.ndarray,
    freq: float,
    weightings: np.ndarray,
    dropout: np.ndarray,
) -> np.ndarray:
    result = np.zeros((X.shape[1], ), dtype=np.float64)
    sin_w = np.sin(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    cos_w = np.cos(np.pi * np.arange(X.shape[1])/(freq*(X.shape[1]-1)))
    for i in range(weightings.shape[0]):
        tmp = np.ones((X.shape[1], ), dtype=np.float64)
        for k, extended_letter in enumerate(word):
            if k > 0:
                tmp = np.roll(tmp, 1)
                tmp[0] = 0
            for letter, occurence in enumerate(extended_letter):
                if occurence > 0:
                    for _ in range(occurence):
                        tmp = tmp * X[letter, :]
                elif occurence < 0:
                    for _ in range(-occurence):
                        tmp = tmp / X[letter, :]
            for _ in range(weightings[i, 2*k+1]):
                tmp = tmp * sin_w
            for _ in range(weightings[i, 2*k+2]):
                tmp = tmp * cos_w
            tmp[dropout[k]] = 0
            tmp = np.cumsum(tmp)
        if weightings.shape[1] == 2*len(word) + 3:
            for _ in range(weightings[i, 2*len(word)+1]):
                tmp = tmp * sin_w
            for _ in range(weightings[i, 2*len(word)+2]):
                tmp = tmp * cos_w
        result += weightings[i, 0] * tmp
    return result


@numba.njit(
    "f8[:,:](f8[:,:], f8[:,:], f8[:], f8[:,:])",
    fastmath=True,
    cache=True,
)
def _ffn(
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    Y = np.zeros((A.shape[0], X.shape[1]))
    for j in range(X.shape[1]):
        for d in range(A.shape[0]):
            Y[d, j] = np.sum(A[d, :] * X[:, j]) + b[d]
    Y = Y * (Y > 0)
    Z = np.zeros((C.shape[0], X.shape[1]))
    for j in range(X.shape[1]):
        for d in range(C.shape[0]):
            Z[d, j] = np.sum(C[d, :] * Y[:, j])
    return Z


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], i4[:,:], "
              "f8[:,:,:], f8[:,:], f8[:,:,:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _ffn_coswiss(
    X: np.ndarray,
    word: np.ndarray,
    freqs: np.ndarray,
    weightings: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    result = np.zeros((freqs.size, X.shape[0], X.shape[2]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for f in range(len(freqs)):
            result[f, i] = _coswiss_single(
                _ffn(X[i], A[f], b[f], C[f]), word, freqs[f], weightings
            )
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], i4[:,:], i4[:,:,:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _leaky_coswiss(
    X: np.ndarray,
    word: np.ndarray,
    freqs: np.ndarray,
    weightings: np.ndarray,
    dropout: np.ndarray,
) -> np.ndarray:
    result = np.zeros((freqs.size, X.shape[0], X.shape[2]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for f in range(len(freqs)):
            result[f, i] = _leaky_coswiss_single(
                X[i], word, freqs[f], weightings, dropout[f]
            )
    return result


@numba.njit(
    "f8[:,:,:](f8[:,:,:], i4[:,:], f4[:], i4[:,:])",
    fastmath=True,
    cache=True,
    parallel=True,
)
def _coswiss(
    X: np.ndarray,
    word: np.ndarray,
    freqs: np.ndarray,
    weightings: np.ndarray,
) -> np.ndarray:
    result = np.zeros((freqs.size, X.shape[0], X.shape[2]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for f in range(len(freqs)):
            result[f, i] = _coswiss_single(X[i], word, freqs[f], weightings)
    return result


class CosWISS(ISS):
    """The Cosine Weighted ISS is a regular ISS over the
    :class:`~fruits.iss.semiring.Reals` semiring with squared
    cosine weighted summands, e.g.::

        cos(pi * |i-j| / (f*N))**s * x_i * x_j

    Args:
        freqs (list of floats): Frequencies ``f`` to calculate the
            weighted ISS for.
        words (Sequence of Words): Words to calculate the ISS for.
            Currently only words of length 2 and 3 are supported.
        exponent (int, optional): Exponent ``s`` of the cosine.
            Defaults to 2.
        total_weighting (bool, optional): If set to true, the outer sum
            also weighted by ``cos(pi*|i_k-N|/(f*N))``, where ``k`` is
            the length of the word. Defaults to false.
        ffn_size (int, optional): If set to an integer, a prior
            two-layer neural network with randomized weights is used to
            transform the input time series at each time step. The
            weights are drawn for each word and frequency independently.
            The given integer is the size of the hidden dimension. Input
            and output dimension are equal.
        dropout (float, optional): If a float between 0 and 1 is given,
            some part of the time series gets set to zero before a
            calculation of a cumulative sum. The given float is the
            probability of one index being set to zero for each extended
            letter in a word.
    """

    def __init__(
        self,
        words: Sequence[Word],
        freqs: Sequence[float],
        exponent: int = 2,
        total_weighting: bool = False,
        ffn_size: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__(words)
        self._total_weighting = total_weighting
        self._freqs = freqs
        for word in words:
            if not isinstance(word, SimpleWord):
                raise ValueError("CosWISS only implemented for simple words")
        self._exponent = exponent
        self._ffn_size = ffn_size
        self._dropout = dropout

    def n_iterated_sums(self) -> int:
        """Total number of iterated sums the current ISS configuration
        produces.
        """
        return len(self._freqs) * len(self.words)

    @property
    def requires_fitting(self) -> bool:
        return self._ffn_size is not None or self._dropout is not None

    def _fit(self, X: np.ndarray) -> None:
        if (d := self._ffn_size) is not None:
            self._A = np.random.random(
                (len(self.words), len(self._freqs), d, X.shape[1])
            )
            self._b = np.random.random((len(self.words), len(self._freqs), d))
            self._C = np.random.random(
                (len(self.words), len(self._freqs), X.shape[1], d)
            )
        if (d := self._dropout) is not None:
            rate = int(d * X.shape[2])
            self._dropout_indices = np.array([
                [[np.random.choice(X.shape[2], size=(rate, ), replace=False)
                  for _ in range(max(map(len, self.words)))]
                 for _ in range(len(self._freqs))]
                for _ in range(len(self.words))
            ], dtype=np.int32)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        result = self.batch_transform(X, batch_size=self.n_iterated_sums())
        return next(iter(result))

    def _get_weightings(self, word: Word) -> np.ndarray:
        p = len(word) + 1 if self._total_weighting else len(word)
        trig_id = []
        trig_exp = [self._exponent, 0]
        trig_coeff = 1
        for k in range(self._exponent+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (self._exponent - k) // (k + 1)
        weightings = np.zeros(
            ((self._exponent+1)**(p-1), 2*p+1),
            dtype=np.int32,
        )
        weightings[:, 0] = 1
        for c, comb in enumerate(product(trig_id, repeat=p-1)):
            for i in range(p-1):
                weightings[c, 0] *= int(comb[i][0])
                weightings[c, 2*i+1] += int(comb[i][1])
                weightings[c, 2*i+3] += int(comb[i][1])
                weightings[c, 2*i+2] += int(comb[i][2])
                weightings[c, 2*i+4] += int(comb[i][2])
        return weightings

    def batch_transform(
        self,
        X: np.ndarray,
        batch_size: int = 1,
    ) -> Generator[np.ndarray, None, None]:
        """Yields a number of iterated sums one after another determined
        by the supplied ``batch_size``.

        Args:
            X (np.ndarray): Univariate or multivariate time series as a
                numpy array of shape
                ``(n_series, n_dimensions, series_length)``.
            batch_size (int, optional): Number of words for which the
                iterated sums are returned at once. Default is 1.
        """
        results = []
        for w, word in enumerate(self.words):
            if self._ffn_size is not None:
                results.append(_ffn_coswiss(
                    X,
                    np.array(list(word), dtype=np.int32),
                    np.array(self._freqs, dtype=np.float32),
                    self._get_weightings(word),
                    self._A[w], self._b[w], self._C[w]
                ))
            elif self._dropout is not None:
                results.append(_leaky_coswiss(
                    X,
                    np.array(list(word), dtype=np.int32),
                    np.array(self._freqs, dtype=np.float32),
                    self._get_weightings(word),
                    self._dropout_indices[w],
                ))
            else:
                results.append(_coswiss(
                    X,
                    np.array(list(word), dtype=np.int32),
                    np.array(self._freqs, dtype=np.float32),
                    self._get_weightings(word),
                ))
            if len(results) == batch_size:
                yield np.concatenate(results, dtype=results[0].dtype)
                results = []
        if len(results) != 0:
            yield np.concatenate(results, dtype=results[0].dtype)

    def _copy(self) -> "CosWISS":
        return CosWISS(
            freqs=self._freqs,
            words=self.words,
            exponent=self._exponent,
            total_weighting=self._total_weighting,
            ffn_size=self._ffn_size,
            dropout=self._dropout,
        )

    def _label(self, index: int) -> str:
        d, r = divmod(index, len(self._freqs))
        string = str(self.words[d])
        string += f"!{self._freqs[r]} : ^{self._exponent}"
        if self._total_weighting:
            string += " : total"
        return string
