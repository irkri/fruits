from typing import Iterator, Literal, Optional, Sequence, Union, overload

import numba
import numpy as np

from fruits.words.word import Word, SimpleWord


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


class CachePlan:
    """Class that creates a plan for the efficient calculation of
    iterated sums using the given words. This plan is needed when the
    mode of an :class:`~fruits.signature.iss.SignatureCalculator` is set
    to "extended". The plan removes repetition in calculation.
    """

    def __init__(self, words: Sequence[Word]) -> None:
        self._words = words
        self._create_plan()

    def _create_plan(self) -> None:
        # prefix search in the word strings generates a cache plan
        self._plan = []
        word_strings = [str(word) for word in self._words]
        for i, wstr in enumerate(word_strings):
            els = wstr.split("[")[1:]
            # start variable used for speed up
            start = 0
            depth = len(els)
            for j in range(len(els)):
                for k in range(start, i):
                    if word_strings[k].startswith("["+"[".join(els[:j+1])):
                        # a prefix in word k is found -> ignore the prefix
                        start = k
                        depth -= 1
                        break
                else:
                    # one prefix didn't exist -> next larger prefix will
                    # not exist either
                    break
            self._plan.append(depth)

    def unique_el_depth(self, index: int) -> int:
        """Returns the total number of iterated sums to be calculated
        for the word with the given index.
        """
        return self._plan[index]

    def n_iterated_sums(self, word_indices: Sequence[int]) -> int:
        """Returns the number of iterated sums that have to be
        calculated using this plan for the words with the given indices.
        """
        return sum(self._plan[i] for i in word_indices)


def _transform_simple_word(word: SimpleWord) -> np.ndarray:
    # transforms the given simple word for faster calculation with a
    # backend function
    simple_word_raw = [el for el in word]
    word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
    for i in range(len(simple_word_raw)):
        for j in range(len(simple_word_raw[i])):
            word_transformed[i, j] = simple_word_raw[i][j]
    return word_transformed


def _ISS(
    X: np.ndarray,
    words: Sequence[Word],
    mode: Literal["single", "extended"] = "single",
    batch_size: Optional[int] = None,
) -> Iterator[np.ndarray]:
    batch_size = len(words) if batch_size is None else batch_size

    cache_plan = CachePlan(words)
    wi = 0
    while wi < len(words):
        n_iterated_sums = batch_size if mode == "single" else (
            cache_plan.n_iterated_sums(list(range(wi, wi+batch_size)))
        )
        results = np.zeros((X.shape[0], n_iterated_sums, X.shape[2]))
        index = 0
        for i in range(wi, wi+batch_size):
            if i > len(words):
                wi = len(words)
                yield results[:, :i, :]
                break
            ext = 1 if mode == "single" else cache_plan.unique_el_depth(i)
            alphas = np.array([0.0] + words[i].alpha + [0.0], dtype=np.float32)
            if isinstance(words[i], SimpleWord):
                results[:, index:index+ext, :] = _fast_ISS(
                    X,
                    _transform_simple_word(words[i]),  # type: ignore
                    alphas,
                    ext,
                )
            elif isinstance(words[i], Word):
                results[:, index:index+ext, :] = _slow_ISS(
                    X,
                    words[i],
                    alphas,
                    ext,
                )
            else:
                raise TypeError(f"Unknown word type: {type(words[i])}")
            index += ext
        else:
            wi += batch_size
            yield results


@overload
def ISS(
    X: np.ndarray,
    words: Union[Sequence[Word], Word],
    mode: Literal["single", "extended"] = "single",
    batch_size: None = None,
) -> np.ndarray:
    ...


@overload
def ISS(
    X: np.ndarray,
    words: Union[Sequence[Word], Word],
    mode: Literal["single", "extended"] = "single",
    batch_size: int = 0,
) -> Iterator[np.ndarray]:
    ...


def ISS(
    X: np.ndarray,
    words: Union[Sequence[Word], Word],
    mode: Literal["single", "extended"] = "single",
    batch_size: Optional[int] = None,
) -> Union[np.ndarray, Iterator[np.ndarray]]:
    """Takes in a number of time series and a list of words and
    calculates the iterated sums for each time series in ``X``.

    Args:
        X (np.ndarray): Three dimensional numpy array of shape
            ``(number of series, dimensions, length of series)``
            containing a multidimensional time series dataset.
        words (Word or list of Words): Words to calculate the ISS for.
        mode (str, optional): Mode of the used calculator. Has to be
            either "single" or "extended".

            - 'single':
                Calculates one iterated sum for each given word.
            - 'extended':
                For each given word, the iterated sum for
                each sequential combination of extended letters
                in that word will be calculated. So for a simple
                word like ``[21][121][1]`` the calculator
                returns the iterated sums for ``[21]``,
                ``[21][121]`` and ``[21][121][1]``.
        batch_size (int, optional): Batch size of the calculator.
            Number of words for which the iterated sums are calculated
            at once when starting the iterator of this calculator. This
            doesn't have to be the same number as the actual number of
            iterated sums returned. Default is that iterated sums for
            all words are given at once.

    Returns
        Numpy array of shape ``(X.shape[0], len(words), X.shape[2])``.
    """
    words = [words] if isinstance(words, Word) else words

    result_iterator = _ISS(
        X,
        words=words,
        mode=mode,
        batch_size=batch_size,
    )
    if batch_size is None:
        return next(iter(result_iterator))
    else:
        return result_iterator
