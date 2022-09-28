import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_cache_plan():
    words = [
        fruits.words.SimpleWord("[1][11][3][11]"),
        fruits.words.SimpleWord("[11][13][11][1][3]"),
        fruits.words.SimpleWord("[1][13][1]"),
        fruits.words.SimpleWord("[11][13][111][13][11]"),
        fruits.words.SimpleWord("[3][11][111]"),
        fruits.words.SimpleWord("[1][11][2]",),
        fruits.words.SimpleWord("[11][2]"),
        fruits.words.SimpleWord("[11][13][111][13][2]"),
        fruits.words.SimpleWord("[3][11][1112][21]"),
    ]

    cache_plan = fruits.iss.CachePlan(words)

    assert cache_plan._plan == [4, 5, 2, 3, 3, 1, 1, 1, 2]


def test_extented_mode_simple_1():
    X = np.random.random_sample((10, 3, 100))

    word = fruits.words.SimpleWord("[11][21][331][22]")

    result_extended = fruits.ISS(X, word, mode="extended")

    word1 = fruits.words.SimpleWord("[11]")
    word2 = fruits.words.SimpleWord("[11][12]")
    word3 = fruits.words.SimpleWord("[11][12][133]")
    word4 = fruits.words.SimpleWord("[11][12][133][22]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.ISS(X, word1, mode="single")
    result_single[:, 1:2, :] = fruits.ISS(X, word2, mode="single")
    result_single[:, 2:3, :] = fruits.ISS(X, word3, mode="single")
    result_single[:, 3:4, :] = fruits.ISS(X, word4, mode="single")

    np.testing.assert_allclose(result_single, result_extended)


def test_extended_mode_simple_2():
    X = np.random.random_sample((10, 3, 100))

    word = fruits.words.SimpleWord("[1][11][111][1111]")

    result_extended = fruits.ISS(X, word, mode="extended")

    word1 = fruits.words.SimpleWord("[1]")
    word2 = fruits.words.SimpleWord("[1][11]")
    word3 = fruits.words.SimpleWord("[1][11][111]")
    word4 = fruits.words.SimpleWord("[1][11][111][1111]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.ISS(X, word1, mode="single")
    result_single[:, 1:2, :] = fruits.ISS(X, word2, mode="single")
    result_single[:, 2:3, :] = fruits.ISS(X, word3, mode="single")
    result_single[:, 3:4, :] = fruits.ISS(X, word4, mode="single")

    np.testing.assert_allclose(result_single, result_extended)


def test_extended_mode_simple_3():
    X = np.random.random_sample((10, 3, 100))

    words = [
        fruits.words.SimpleWord("[1][11][3][11]"),
        fruits.words.SimpleWord("[11][13][11][1][3]"),
        fruits.words.SimpleWord("[1][13][1]"),
        fruits.words.SimpleWord("[11][13][111][13][11]"),
        fruits.words.SimpleWord("[3][11][111]"),
        fruits.words.SimpleWord("[1][11][2]",),
        fruits.words.SimpleWord("[11][2]"),
        fruits.words.SimpleWord("[11][13][111][13][2]"),
        fruits.words.SimpleWord("[3][11][1112][21]"),
    ]

    all_words = [
        fruits.words.SimpleWord("[1]"),
        fruits.words.SimpleWord("[1][11]"),
        fruits.words.SimpleWord("[1][11][3]"),
        fruits.words.SimpleWord("[1][11][3][11]"),
        fruits.words.SimpleWord("[11]"),
        fruits.words.SimpleWord("[11][13]"),
        fruits.words.SimpleWord("[11][13][11]"),
        fruits.words.SimpleWord("[11][13][11][1]"),
        fruits.words.SimpleWord("[11][13][11][1][3]"),
        fruits.words.SimpleWord("[1][13]"),
        fruits.words.SimpleWord("[1][13][1]"),
        fruits.words.SimpleWord("[11][13][111]"),
        fruits.words.SimpleWord("[11][13][111][13]"),
        fruits.words.SimpleWord("[11][13][111][13][11]"),
        fruits.words.SimpleWord("[3]"),
        fruits.words.SimpleWord("[3][11]"),
        fruits.words.SimpleWord("[3][11][111]"),
        fruits.words.SimpleWord("[1][11][2]",),
        fruits.words.SimpleWord("[11][2]"),
        fruits.words.SimpleWord("[11][13][111][13][2]"),
        fruits.words.SimpleWord("[3][11][1112]"),
        fruits.words.SimpleWord("[3][11][1112][21]"),
    ]

    result_extended = fruits.ISS(X, words, mode="extended")

    result_single = fruits.ISS(X, all_words, mode="single")

    np.testing.assert_allclose(result_single, result_extended)


def test_extended_mode_general_1():
    X = np.random.random_sample((10, 3, 100))

    if "ReLU" not in fruits.words.letters.get_available():
        @fruits.words.letter(name="ReLU")
        def relu(X, i):
            return X[i, :] * (X[i, :] > 0)

    if "EXP" not in fruits.words.letters.get_available():
        @fruits.words.letter(name="EXP")
        def exp(X, i):
            return np.exp(-X[i, :] / 100)

    el1 = fruits.words.ExtendedLetter("ReLU(0)ReLU(0)")
    el2 = fruits.words.ExtendedLetter("ReLU(1)EXP(0)")
    el3 = fruits.words.ExtendedLetter("EXP(1)EXP(2)")

    relu_word = fruits.words.Word()
    relu_word.multiply(el1)
    relu_word.multiply(el2)
    relu_word.multiply(el3)

    relu_word1 = fruits.words.Word()
    relu_word1.multiply(el1)

    relu_word2 = fruits.words.Word()
    relu_word2.multiply(el1)
    relu_word2.multiply(el2)

    relu_word3 = fruits.words.Word()
    relu_word3.multiply(el1)
    relu_word3.multiply(el2)
    relu_word3.multiply(el3)

    result_extended = fruits.ISS(X, relu_word, mode="extended")

    result_single = np.zeros((10, 3, 100))

    result_single[:, 0:1, :] = fruits.ISS(X, relu_word1, mode="single")
    result_single[:, 1:2, :] = fruits.ISS(X, relu_word2, mode="single")
    result_single[:, 2:3, :] = fruits.ISS(X, relu_word3, mode="single")

    np.testing.assert_allclose(result_single, result_extended)
