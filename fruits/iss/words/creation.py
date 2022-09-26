import itertools
from collections.abc import Iterator, Sequence
from typing import Union

from .letters import ExtendedLetter
from .word import SimpleWord, Word


def _partitions_of(n, start: int = 1):
    yield (n,)
    for i in range(start, n//2 + 1):
        for p in _partitions_of(n-i, i):
            yield (i,) + p


def _extended_letters_by_weight(
    w: int,
    d: int = 1
) -> list[str]:
    return [
        "["
        + "".join(["(" + str(x) + ")" if len(str(x)) > 1 else str(x)
                   for x in el])
        + "]"
        for el in
        itertools.combinations_with_replacement(list(range(1, d+1)), w)
    ]


def of_weight(w: int, dim: int = 1) -> tuple[SimpleWord, ...]:
    """Returns a list of all possible and unique words that have exactly
    the given number of (simple) letters ('weight' of the words).
    For ``w=2`` and ``dim=2`` this will return a list containing::

        SimpleWord("[11]"), SimpleWord("[12]"), SimpleWord("[22]"),
        SimpleWord("[1][1]"), SimpleWord("[1][2]"),
        SimpleWord("[2][1]"), SimpleWord("[2][2]")

    Args:
        w (int): Weight of the words, i.e. number of letters.
        dim (int, optional): Highest dimension of a letter used.
            Defaults to 1.
    """
    extended_letters = []
    words = []
    for i in range(1, w+1):
        extended_letters.append(_extended_letters_by_weight(i, dim))
    for partition in _partitions_of(w):
        for mixed_up_partition in set(itertools.permutations(partition)):
            raw_words = itertools.product(*[extended_letters[weight-1]
                                            for weight in mixed_up_partition])
            for raw_word in raw_words:
                words.append(SimpleWord("".join(raw_word)))
    return tuple(words)


def replace_letters(
    words: Union[SimpleWord, Sequence[SimpleWord]],
    letter_gen: Iterator[str],
) -> Union[Word, tuple[Word, ...]]:
    """Replaces the letters in the given simple word(s) by the letters
    specified as strings in ``letter_gen``.

    Args:
        word (Word or sequence of Words): Words with letters to replace.
        letter_gen (Generator): Iterator that yields letter names of
            correctly decorated functions
            (using `meth:`~fruits.words.letters.letter``). If the
            iteration through the generator is stopped, all left letters
            in the word will not be changed.

    Returns:
        Word or list of Words based on the input.
    """
    if isinstance(words, SimpleWord):
        words = (words, )
    if not isinstance(words, Sequence):
        raise ValueError("'Invalid argument for 'words' specified")
    new_words = []
    for i in range(len(words)):
        if not isinstance(words[i], SimpleWord):
            raise ValueError("Can only replace letters in a simple word")
        new_word = Word()
        for el in words[i]:
            new_el = ExtendedLetter()
            for dim in el._dimensions:
                for _ in range(dim):
                    try:
                        letter = next(letter_gen)
                    except StopIteration:
                        letter = "DIM"
                    new_el.append(letter, dim)
            new_word.multiply(new_el)
        new_words.append(new_word)
    return tuple(new_words)
