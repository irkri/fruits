from typing import Iterator, Literal, Optional, Sequence, Union, overload

import numpy as np

from .words.word import Word, SimpleWord
from ._backend import _fast_ISS, _slow_ISS


class CachePlan:
    """Class that creates a plan for the efficient calculation of
    iterated sums using the given words. This plan is needed when the
    mode of an :meth:`~fruits.iss.ISS` calculation is set to "extended".
    The plan removes repetition in calculation.
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
    simple_word_raw = list(word)
    word_transformed = np.zeros((len(word), word._max_dim), dtype=np.int32)
    for i, dim in enumerate(simple_word_raw):
        for j, nletters in enumerate(dim):
            word_transformed[i, j] = nletters
    return word_transformed


def _ISS(
    X: np.ndarray,
    words: Sequence[Word],
    mode: Literal["single", "extended"],
    batch_size: int,
) -> Iterator[np.ndarray]:
    if batch_size > len(words):
        raise ValueError("batch_size too large, has to be < len(words)")

    cache_plan = CachePlan([])
    if mode == "extended":
        cache_plan = CachePlan(words)

    i = 0
    while i < len(words):
        if i + batch_size > len(words):
            batch_size = len(words) - i
        results = np.zeros((
            X.shape[0],
            batch_size if mode == "single" else (
                cache_plan.n_iterated_sums(range(i, i+batch_size))
            ),
            X.shape[2],
        ))
        index = 0
        for word in words[i:i+batch_size]:
            ext = 1 if mode == "single" else cache_plan.unique_el_depth(i)
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
        batch_size=batch_size if batch_size is not None else len(words),
    )
    if batch_size is None:
        return next(iter(result_iterator))
    return result_iterator
