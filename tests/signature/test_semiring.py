import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2.0, 1, 0, 0, -7]],
    [[5.0, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])

def test_simpleword_arctic():
    w1 = fruits.words.SimpleWord("[1]")
    w2 = fruits.words.SimpleWord("[2]")
    w3 = fruits.words.SimpleWord("[11]")
    w4 = fruits.words.SimpleWord("[12]")
    w5 = fruits.words.SimpleWord("[1][1]")
    w6 = fruits.words.SimpleWord("[1][2]")

    correct = (
        np.array([[-4, 0.8, 0.8, 5, 5], [5, 8, 8, 8, 8]]),
        np.array([[2, 2, 2, 2, 2], [-5, -1, -1, -0.5, -0.5]]),
        np.array([[-8, 1.6, 1.6, 10, 10], [10, 16, 16, 16, 16]]),
        np.array([[-2, 1.8, 1.8, 5, 5], [0, 7, 7, 7, 7]]),
        np.array([[-8, 1.6, 1.6, 10, 10], [10, 16, 16, 16, 16]]),
        np.array([[-2, 1.8, 1.8, 5., 5.], [0., 7., 7., 7.5, 7.5]]),
    )

    results = fruits.ISS(
        [w1, w2, w3, w4, w5, w6],
        semiring=fruits.iss.semiring.Arctic(),
    ).batch_transform(X_1, batch_size=1)

    for i, result in enumerate(results):
        np.testing.assert_allclose(correct[i], result[0])


def test_word_arctic():
    word = fruits.words.Word("[DIM(1)DIM(2)][DIM(1)ABS(1)]")

    correct = np.array([[-2, 3.4, 3.4, 15, 15], [10, 23, 23, 23, 23]])

    results = fruits.ISS(
        (word, ),
        semiring=fruits.iss.semiring.Arctic(),
    ).fit_transform(X_1)

    np.testing.assert_allclose(correct, results[0, :, :])
