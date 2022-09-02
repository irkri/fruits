import itertools
from collections.abc import Iterator, Sequence
from typing import Union

from fruits.words.letters import (
    FREE_LETTER_TYPE,
    ExtendedLetter,
    _is_letter,
    simple
)
from fruits.words.word import SimpleWord, Word


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


def _replace_letters_simpleword(word, letter_gen):
    complexword = Word()
    for el in word:
        new_el = ExtendedLetter()
        for dim, ndim in enumerate(el):
            for _ in range(ndim):
                try:
                    letter = next(letter_gen)
                except StopIteration:
                    letter = simple
                if not _is_letter(letter):
                    raise TypeError("Letter has the wrong signature. "
                                    "Perhaps it wasn't decorated "
                                    "correctly?")
                new_el.append(letter, dim)
        complexword.multiply(new_el)
    return complexword


def _replace_letters_complexword(word, letter_gen):
    complexword = Word()
    for el in word:
        new_el = ExtendedLetter()
        for i, dim in zip(el._letters, el._dimensions):
            try:
                letter = next(letter_gen)
            except StopIteration:
                letter = i
            if not _is_letter(letter):
                raise TypeError("Letter has the wrong signature. "
                                + "Perhaps it wasn't decorated "
                                + "correctly?")
            new_el.append(letter, dim)
        complexword.multiply(new_el)
    return complexword


def replace_letters(
    word: Union[Word, Sequence[Word]],
    letter_gen: Iterator[FREE_LETTER_TYPE],
) -> Union[Word, tuple[Word, ...]]:
    """Replaces the letters in the given word(s) by the iteration
    results from the supplied generator.

    Args:
        word (Word or sequence of Words): Words with letters to replace.
        letter_gen (Generator): Iterator that yields functions correctly
            decorated with :meth:`~fruits.words.letters.letter``. If the
            iteration through the generator is stopped, all left letters
            in the word will not be changed.

    Returns:
        Word or list of Words based on the input.
    """
    if isinstance(word, Sequence):
        complexwords = []
        for i in range(len(word)):
            if isinstance(word[i], SimpleWord):
                complexwords.append(
                    _replace_letters_simpleword(word[i], letter_gen)
                )
            elif isinstance(word[i], Word):
                complexwords.append(
                    _replace_letters_complexword(word[i], letter_gen)
                )
            else:
                raise TypeError(f"Unknown word type: {type(word[i])}")
        return tuple(complexwords)
    else:
        if isinstance(word, SimpleWord):
            return _replace_letters_simpleword(word, letter_gen)
        elif isinstance(word, Word):
            return _replace_letters_complexword(word, letter_gen)
        else:
            raise TypeError(f"Unknown word type: {type(word)}")
