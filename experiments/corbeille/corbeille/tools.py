from typing import Literal, Union

import fruits

import numpy as np


def split_index(
    fruit: fruits.Fruit,
    index: int,
    level: Literal["prepared", "iterated sums", "features"] = "features",
) -> tuple[int, ...]:
    """For a given index at the specified level, returns the indices
    needed to access the corresponding element in the fruit.

    Args:
        index (int): The index to decode.
        level (str, optional): Stage of the transformation that is
            indexed. Possible levels are
            ``['prepared', 'iterated sums', 'features']``.
            Defaults to ``'features'``.

    Returns:
        A tuple of integer indices. The length of the tuple is
        dependent on the level chosen. The indices in order
        correspond to ``(slice, word, sieve, feature)`` as one sieve can
        output many features.
    """
    if level == "prepared":
        for slc_index in range(len(fruit)):
            if index == 0:
                return (slc_index, )
            index -= 1
    elif level == "iterated sums":
        for slc_index, slc in enumerate(fruit):
            nwords = np.prod([iss.n_iterated_sums() for iss in slc.get_iss()])
            for word_index in range(nwords):
                if index == 0:
                    return (slc_index, word_index)
                index -= 1
    elif level == "features":
        for slc_index, slc in enumerate(fruit):
            nwords = np.prod([iss.n_iterated_sums() for iss in slc.get_iss()])
            for word_index in range(nwords):
                for s_index, sieve in enumerate(slc.get_sieves()):
                    for feature_index in range(sieve.nfeatures()):
                        if index == 0:
                            return (
                                slc_index, word_index, s_index, feature_index,
                            )
                        index -= 1
    raise ValueError("Index out of range or unknown level")
