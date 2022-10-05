import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_simpleword_minplus():
    w1 = fruits.words.SimpleWord("[1]")
    w2 = fruits.words.SimpleWord("[2]")
    w3 = fruits.words.SimpleWord("[11]")
    w4 = fruits.words.SimpleWord("[12]")
    w5 = fruits.words.SimpleWord("[1][1]")
    w6 = fruits.words.SimpleWord("[1][2]")

    correct = (
        np.array([[-4, -4, -4, -4, -4], [5, 5, 2, 2, 0]]),
        np.array([[2, 1, 0, 0, -7], [-5, -5, -5, -5, -8]]),
        np.array([[-8, -8, -8, -8, -8], [10, 10, 4, 4, 0]]),
        np.array([[-2, -2, -2, -2, -10], [0, 0, -2, -2, -8]]),
        np.array([[0, -3.2, -4, -4, -7], [0, 13, 7, 7, 2]]),
        np.array([[0, -3, -4, -4, -11], [0, 4, 1, 1, -6]]),
    )

    results = fruits.ISS(
        [w1, w2, w3, w4, w5, w6],
        semiring=fruits.iss.semiring.MinPlus(),
    ).batch_transform(X_1, batch_size=1)

    for i, result in enumerate(results):
        np.testing.assert_allclose(correct[i], result[:, 0, :])


def test_word_minplus():
    word = fruits.words.Word("[DIM(1)DIM(2)][DIM(1)ABS(1)]")

    correct = np.array([[0, -0.4, -2, -2, -2], [0, 16, 4, 4, -2]])

    results = fruits.ISS(
        (word, ),
        semiring=fruits.iss.semiring.MinPlus(),
    ).fit_transform(X_1)

    np.testing.assert_allclose(correct, results[:, 0, :])
