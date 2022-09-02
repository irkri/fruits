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
        ``(branch, word, [extended letter,] sieve, feature)`` as one
        sieve can output many features. The ``extended letter`` will
        only be produced if the corresponding fruit branch has the
        ``iss_mode ``set to 'extended'.
    """
    if level == "prepared":
        for branch_index in range(len(fruit.branches())):
            if index == 0:
                return (branch_index, )
            index -= 1
    elif level == "iterated sums":
        for branch_index, branch in enumerate(fruit.branches()):
            cp = None if branch.iss_mode != "extended" else (
                CachePlan(branch.get_words())
            )
            for word_index, word in enumerate(branch.get_words()):
                n = cp.unique_el_depth(word_index) if cp is not None else 1
                for ext_letter in range(len(word)-n, len(word)):
                    if index == 0:
                        return (
                            (branch_index, word_index, ext_letter)
                            if branch.iss_mode == "extended"
                            else (branch_index, word_index)
                        )
                    index -= 1
    elif level == "features":
        for branch_index, branch in enumerate(fruit.branches()):
            cp = None if branch.iss_mode != "extended" else (
                CachePlan(branch.get_words())
            )
            for word_index, word in enumerate(branch.get_words()):
                n = cp.unique_el_depth(word_index) if cp is not None else 1
                for ext_letter in range(len(word)-n, len(word)):
                    for s_index, sieve in enumerate(branch.get_sieves()):
                        for feature_index in range(sieve.nfeatures()):
                            if index == 0:
                                return (
                                    (branch_index, word_index, ext_letter,
                                     s_index, feature_index)
                                    if branch.iss_mode == "extended"
                                    else (branch_index, word_index, s_index,
                                          feature_index)
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
    branch = fruit.branch(index[0])
    if with_kwargs:
        string = "->".join(map(str, branch.get_preparateurs()))
    else:
        string = "->".join(map(
            lambda x: str(x).split("(")[0],
            branch.get_preparateurs()
        ))
    if level == "iterated sums" or level == "features":
        if string != "":
            string += "->"
        if branch.iss_mode == "extended":
            string += "]".join(
                str(branch.get_words()[index[1]]).split("]")[:-1][:index[2]+1]
            ) + "]"
        else:
            string += str(branch.get_words()[index[1]])
    if level == "features":
        string += "->"
        if with_kwargs:
            string += str(branch.get_sieves()[index[-2]])
        else:
            string += str(branch.get_sieves()[index[-2]]).split("(")[0]
        string += f"_{index[-1]}"
    elif level != "prepared" and level != "iterated sums":
        raise ValueError(f"Unknown level supplied: {level!r}")
    return string
