import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_fast_slow_iss():
    @fruits.words.letter
    def dim_letter(X, i):
        return X[i, :]

    @fruits.words.letter(name="single_dim_letter")
    def second_dim_letter(X, i):
        return X[i, :]

    # word [11122][122222][11]
    el_1_1 = fruits.words.ExtendedLetter()
    for _ in range(3):
        el_1_1.append("dim_letter", 0)
    for _ in range(2):
        el_1_1.append("dim_letter", 1)

    el_1_2 = fruits.words.ExtendedLetter()
    for _ in range(1):
        el_1_2.append("dim_letter", 0)
    for _ in range(5):
        el_1_2.append("dim_letter", 1)

    el_1_3 = fruits.words.ExtendedLetter()
    for _i in range(2):
        el_1_3.append("dim_letter", 0)

    word1 = fruits.words.Word("Word 1")
    word1.multiply(el_1_1)
    word1.multiply(el_1_2)
    word1.multiply(el_1_3)

    # word [22][112][2221]
    el_2_1 = fruits.words.ExtendedLetter()
    for _ in range(2):
        el_2_1.append("single_dim_letter", 1)

    el_2_2 = fruits.words.ExtendedLetter()
    for _ in range(2):
        el_2_2.append("single_dim_letter", 0)
    for _ in range(1):
        el_2_2.append("single_dim_letter", 1)

    el_2_3 = fruits.words.ExtendedLetter()
    for _ in range(1):
        el_2_3.append("single_dim_letter", 0)
    for _ in range(3):
        el_2_3.append("single_dim_letter", 1)

    word2 = fruits.words.Word("Word 2")
    word2.multiply(el_2_1)
    word2.multiply(el_2_2)
    word2.multiply(el_2_3)

    sit1 = fruits.words.SimpleWord("[11122][122222][11]")
    sit2 = fruits.words.SimpleWord("[22][112][2221]")

    X = np.random.random_sample((100, 2, 100))

    result_fast = fruits.ISS(X, [sit1, sit2])
    result_slow = fruits.ISS(X, [word1, word2])

    np.testing.assert_allclose(result_slow, result_fast)

    word1_copy = word1.copy()
    word2_copy = word2.copy()
    result_slow_copy = fruits.ISS(X, [word1_copy, word2_copy])

    np.testing.assert_allclose(result_slow, result_slow_copy)


def test_general_words():
    if "ReLU" not in fruits.words.letters.get_available():
        @fruits.words.letter(name="ReLU")
        def relu(X, i):
            return X[i, :] * (X[i, :] > 0)

    # word: [relu(0)][relu(1)]
    relu_word = fruits.words.Word("relu collection")
    relu_word.multiply(fruits.words.ExtendedLetter("ReLU(1)"))
    relu_word.multiply(fruits.words.ExtendedLetter("ReLU(2)"))

    mix = [relu_word, fruits.words.SimpleWord("[111]")]

    mix_result = fruits.ISS(X_1, mix)

    np.testing.assert_allclose(np.array([
        [[0, 0, 0, 0, 0],
         [-64, -63.488, -63.488, 61.512, 34.512]],
        [[0, 0, 0, 0, 0],
         [125, 637, 645, 861, 861]]
    ]),
        mix_result)
