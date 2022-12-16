from typing import Literal, Union

import fruits
from fruits.iss import CachePlan


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
        correspond to
        ``(slice, iss, word, [extended letter,] sieve, feature)`` as one
        sieve can output many features. The ``extended letter`` will
        only be produced if the corresponding fruit slice and ISS
        calculator has its ``mode`` set to ``fruits.ISSMode.EXTENDED``.
    """
    if level == "prepared":
        for slc_index in range(len(fruit)):
            if index == 0:
                return (slc_index, )
            index -= 1
    elif level == "iterated sums":
        for slc_index, slc in enumerate(fruit):
            for iss_index, iss in enumerate(slc.get_iss()):
                for word_index, word in enumerate(iss.words):
                    n = 1
                    if iss.mode == fruits.ISSMode.EXTENDED:
                        n = iss._cache_plan.unique_el_depth(word_index)
                    for ext_letter in range(len(word)-n, len(word)):
                        if index == 0:
                            return (
                                (slc_index, iss_index, word_index, ext_letter)
                                if iss.mode == fruits.ISSMode.EXTENDED
                                else (slc_index, iss_index, word_index)
                            )
                        index -= 1
    elif level == "features":
        for slc_index, slc in enumerate(fruit):
            for iss_index, iss in enumerate(slc.get_iss()):
                for word_index, word in enumerate(iss.words):
                    n = 1
                    if iss.mode == fruits.ISSMode.EXTENDED:
                        n = iss._cache_plan.unique_el_depth(word_index)
                    for ext_letter in range(len(word)-n, len(word)):
                        for s_index, sieve in enumerate(slc.get_sieves()):
                            for feature_index in range(sieve.nfeatures()):
                                if index == 0:
                                    return (
                                        (slc_index, iss_index, word_index,
                                         ext_letter, s_index, feature_index)
                                        if iss.mode == fruits.ISSMode.EXTENDED
                                        else (slc_index, iss_index, word_index,
                                              s_index, feature_index)
                                    )
                                index -= 1
    raise ValueError("Index out of range or unknown level")


def transformation_string(
    fruit: fruits.Fruit,
    index: Union[int, tuple[int, ...]],
    level: Literal["prepared", "iterated sums", "features"] = "features",
    with_kwargs: bool = False,
) -> str:
    """Returns a string characterising the transformation needed to get
    to the supplied data result with the given fruit. The string
    consists of seed identifiers within the fruit.

    Args:
        fruit (fruits.Fruit): A fruits feature extractor.
        index (int or tuple of ints): An index of the data that is
            searched for or a tuple of integers identifying the steps in
            the given fruit one after another. A single integer will be
            first transformed to a tuple with :meth.`split_index`.
        level (str, optional):Stage of the transformation that is
            indexed. Possible levels are
            ``['prepared', 'iterated sums', 'features']``.
            Defaults to ``'features'``.
        with_kwargs (bool, optional): Whether to put the keyword
            arguments behind the seeds in the string. If set to false,
            only the seeds class name is shown.
    """
    if isinstance(index, int):
        index = split_index(fruit, index, level=level)
    slc = fruit.get_slice(index[0])
    if with_kwargs:
        string = "->".join(map(str, slc.get_preparateurs()))
    else:
        string = "->".join(map(
            lambda x: str(x).split("(")[0],
            slc.get_preparateurs()
        ))
    if level == "iterated sums" or level == "features":
        iss = slc.get_iss()[index[1]]
        if string != "":
            string += "->"
        if iss.mode == fruits.ISSMode.EXTENDED:
            string += "]".join(
                str(iss.words[index[2]]).split("]")[:-1][:index[3]+1]
            ) + "]"
        else:
            string += str(iss.words[index[2]])
        if not isinstance(iss.semiring, fruits.semiring.Reals):
            string += f":{iss.semiring.__class__.__name__}"
    if level == "features":
        string += "->"
        if with_kwargs:
            string += str(slc.get_sieves()[index[-2]])
        else:
            string += str(slc.get_sieves()[index[-2]]).split("(")[0]
        string += f"_{index[-1]}"
    elif level != "prepared" and level != "iterated sums":
        raise ValueError(f"Unknown level supplied: {level!r}")
    return string
