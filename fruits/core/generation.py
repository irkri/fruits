import itertools

from fruits.core.wording import SimpleWord

def simplewords_by_degree(max_letters: int,
                          max_extended_letters: int,
                          dim: int = 1) -> list:
    """Returns all possible and unique SimpleWords up to the given 
    boundaries.
    For ``max_letters=2``, ``max_extended_letters=2`` and ``dim=1``
    this will return a list containing::

        SimpleWord("[1]"), SimpleWord("[1][1]"), SimpleWord("[11]"),
        SimpleWord("[11][1]"), SimpleWord("[1][11]"),
        SimpleWord("[11][11]")
    
    :param max_letters: Maximal number of letters in any extended
        letter.
    :type monomial_length: int
    :param max_letters: Maximal number of extended letters in any
        SimpleWord.
    :type n_extended_letters: int
    :param dim: Maximal dimensionality of the letters in any extended
        letter and any SimpleWord., defaults to 1
    :type dim: int, optional
    :returns: List of SimpleWord objects.
    :rtype: list
    """
    ext_letters = []
    for l in range(1, max_letters+1):
        for ext_letter in itertools.combinations_with_replacement(
                                        list(range(1, dim+1)), l):
            ext_letters.append(list(ext_letter))
    words = []
    for n in range(1, max_extended_letters+1):
        for word in itertools.product(ext_letters, repeat=n):
            words.append("".join([str(x).replace(", ","") for x in word]))

    print(f"{words=}")
    for i in range(len(words)):
        words[i] = SimpleWord(words[i])

    return words

def simplewords_by_length(l: int,
                          dim: int = 1) -> list:
    """Returns a list of all possible and unique SimpleWords that have
    exactly the given number of letters.
    For ``l=2`` and ``dim=2`` this will return a list containing::

        SimpleWord("[11]"), SimpleWord("[12]"), SimpleWord("[22]"),
        SimpleWord("[1][1]"), SimpleWord("[1][2]"), SimpleWord("[2][2]")
    
    :param l: Number of letters the words should contain
    :type l: int
    :param dim: Highest dimension of a letter., defaults to 1
    :type dim: int, optional
    """
    # generate all extended_letters that can occured in a word with
    # exactly l letters
    extended_letters = []
    for length in range(1, l+1):
        extended_letters.append([])
        mons = itertools.combinations_with_replacement(
                                list(range(1, dim+1)), length)
        for mon in mons:
            extended_letters[-1].append(list(mon))
    # generate all possible combinations of the extended_letters created 
    # above such that the combination is a word with l letters
    # (next line is in O(2^l), maybe find a better option later)
    choose_extended_letters = [t for i in range(1, l+1) for t in
                               itertools.combinations_with_replacement(
                                list(range(1, l+1)), i)
                               if sum(t)==l]
    # use the combinations above for generating all possible words with
    # l letters by using the extended_letters from the beginning
    words = []
    for choice in choose_extended_letters:
        selected_extended_letters = []
        for perm in set(itertools.permutations(choice)):
            selected_extended_letters.append(list(itertools.product(
                *[extended_letters[i-1] for i in perm])))
        for inner_words in [[str(list(x))[1:-1].replace(", ","")
                             for x in mon] for mon in selected_extended_letters]:
            for word in inner_words:
                words.append(SimpleWord(word))
    return words
