from typing import Generator, Optional, Sequence

import numpy as np

from .cache import CachePlan
from .semiring import ISSSemiRing
from .words.word import Word

def calculate_ISS(
    X: np.ndarray,
    words: Sequence[Word],
    batch_size: int,
    semiring: ISSSemiRing,
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
            results[:, index:index+ext, :] = semiring.iterated_sums(
                X,
                word,
                alphas,
                ext,
            )
            index += ext
            i += 1
        yield results
