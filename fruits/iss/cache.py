from typing import Optional, Sequence

from .words.word import Word


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

    def get_word_index(self, is_index: int) -> int:
        """Returns the word index for the given index of an iterated
        sum.
        """
        for i in range(len(self._words)):
            is_index -= self.unique_el_depth(i)
            if is_index < 0:
                return i
        raise IndexError("Not enough iterated sums in cache plan")

    def get_word_string(self, is_index: int) -> str:
        """Returns the word string for the given index of an iterated
        sum. This can be a prefix of a given word.
        """
        for i in range(len(self._words)):
            is_index -= self.unique_el_depth(i)
            if is_index < 0:
                return "]".join(str(self._words[i]).split("]")[
                    :int(is_index)
                ]) + "]"
        raise IndexError("Not enough iterated sums in cache plan")

    def n_iterated_sums(
        self,
        word_indices: Optional[Sequence[int]] = None,
    ) -> int:
        """Returns the number of iterated sums that will be calculated
        using this cache plan.

        Args:
            word_indices (sequence of ints): If a number of word indices
                is supplied, only the corresponding words will be
                considered in the returned number of iterated sums.
        """
        if word_indices is None:
            word_indices = range(len(self._words))
        return sum(self._plan[i] for i in word_indices)
