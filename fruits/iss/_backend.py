from typing import Generator, Optional, Sequence

import numba
import numpy as np

from .cache import CachePlan
from .words.word import SimpleWord, Word


def _slow_single_ISS(
    Z: np.ndarray,
    word: Word,
    alphas: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones(Z.shape[1], dtype=np.float64)
    for k, ext_letter in enumerate(word):
        C = np.ones(Z.shape[1], dtype=np.float64)
        for el in ext_letter:
            C = C * el(Z[:, :])
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        if alphas[k+1] != alphas[k] or alphas[k] != 0:
            tmp = tmp * np.exp(np.arange(Z.shape[1])
                               * (alphas[k+1]-alphas[k])
                               + alphas[k])
        tmp = _fast_CS(tmp)
        if len(word)-k <= extended:
            # save result
            result[extended-(len(word)-k), :] = tmp.copy()
    return result


def _slow_ISS(
    Z: np.ndarray,
    word: Word,
    alphas: np.ndarray,
    extended: int,
) -> np.ndarray:
    # calculates iterated sums of Z for a list of general words
    result = np.ones((Z.shape[0], extended, Z.shape[2]))
    for i in range(Z.shape[0]):
        result[i] = _slow_single_ISS(Z[i], word, alphas, extended)
    return result


@numba.njit("float64[:](float64[:])", cache=True)
def _fast_CS(Z: np.ndarray) -> np.ndarray:
    return np.cumsum(Z)


@numba.njit("float64[:](float64[:, :], int32[:])", cache=True)
def _fast_extended_letter(
    Z: np.ndarray,
    extended_letter: np.ndarray,
) -> np.ndarray:
    C = np.ones(Z.shape[1], dtype=np.float64)
    for dim, el in enumerate(extended_letter):
        if el != 0:
            C = C * Z[dim, :]**el
    return C


@numba.njit(
    "float64[:,:](float64[:,:], int32[:,:], float32[:], int32)",
    fastmath=True,
    cache=True,
)
def _fast_single_ISS(
    Z: np.ndarray,
    word: np.ndarray,
    alphas: np.ndarray,
    extended: int,
) -> np.ndarray:
    result = np.ones((extended, Z.shape[1]), dtype=np.float64)
    tmp = np.ones(Z.shape[1], dtype=np.float64)
    for k, ext_letter in enumerate(word):
        if not np.any(ext_letter):
            continue
        C = _fast_extended_letter(Z, ext_letter)
        if k > 0:
            tmp = np.roll(tmp, 1)
            tmp[0] = 0
        tmp = tmp * C
        if alphas[k+1] != alphas[k] or alphas[k] != 0:
            tmp = tmp * np.exp(np.arange(Z.shape[1])
                               * (alphas[k+1]-alphas[k])
                               + alphas[k])
        tmp = _fast_CS(tmp)
        if len(word)-k <= extended:
            # save result
            result[extended-(len(word)-k), :] = tmp.copy()
    return result


@numba.njit(
    "float64[:,:,:](float64[:,:,:], int32[:,:], float32[:], int32)",
    parallel=True,
    cache=True,
)
def _fast_ISS(
    Z: np.ndarray,
    word: np.ndarray,
    alphas: np.ndarray,
    extended: int,
) -> np.ndarray:
    # accelerated function for calculation of
    # fruits.core.ISS(X, [SimpleWord(...)])
    result = np.zeros((Z.shape[0], extended, Z.shape[2]))
    for i in numba.prange(Z.shape[0]):
        result[i] = _fast_single_ISS(Z[i, :, :], word, alphas, extended)
    return result


def _transform_simple_word(word: SimpleWord) -> np.ndarray:
    # transforms the given simple word for faster calculation with a
    # backend function
    simple_word_raw = list(word)
    word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
    for i, dim in enumerate(simple_word_raw):
        for j, nletters in enumerate(dim):
            word_transformed[i, j] = nletters
    return word_transformed


def calculate_ISS(
    X: np.ndarray,
    words: Sequence[Word],
    batch_size: int,
    cache_plan: Optional[CachePlan] = None,
) -> Generator[np.ndarray, None, None]:
    if batch_size > len(words):
        raise ValueError("batch_size too large, has to be < len(words)")
    i = 0
    while i < len(words):
        if i + batch_size > len(words):
            batch_size = len(words) - i
        results = np.zeros((
            X.shape[0],
            batch_size if cache_plan is None else (
                cache_plan.n_iterated_sums(range(i, i+batch_size))
            ),
            X.shape[2],
        ))
        index = 0
        for word in words[i:i+batch_size]:
            ext = 1 if cache_plan is None else cache_plan.unique_el_depth(i)
            alphas = np.array([0.0] + word.alpha + [0.0], dtype=np.float32)
            if isinstance(word, SimpleWord):
                results[:, index:index+ext, :] = _fast_ISS(
                    X,
                    _transform_simple_word(word),
                    alphas,
                    ext,
                )
            elif isinstance(word, Word):
                results[:, index:index+ext, :] = _slow_ISS(
                    X,
                    word,
                    alphas,
                    ext,
                )
            else:
                raise TypeError(f"Unknown word type: {type(word)}")
            index += ext
            i += 1
        yield results
