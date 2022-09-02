import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_simpleword_iss():
    w1 = fruits.words.SimpleWord("[1]")
    w2 = fruits.words.SimpleWord("[2]")
    w3 = fruits.words.SimpleWord("[11]")
    w4 = fruits.words.SimpleWord("[12]")
    w5 = fruits.words.SimpleWord("[1][1]")
    w6 = fruits.words.SimpleWord("[1][2]")

    r1 = fruits.ISS(X_1, [w1, w2, w3, w4, w5, w6])

    np.testing.assert_allclose(np.array([
        [-4, -3.2, -3.2, 1.8, -1.2],
        [5, 13, 15, 21, 21]
    ]),
        r1[:, 0, :])
    np.testing.assert_allclose(np.array([
        [2, 3, 3, 3, -4],
        [-5, -6, -10, -10.5, -18.5]
    ]),
        r1[:, 1, :])
    np.testing.assert_allclose(np.array([
        [16, 16.64, 16.64, 41.64, 50.64],
        [25, 89, 93, 129, 129]
    ]),
        r1[:, 2, :])
    np.testing.assert_allclose(np.array([
        [-8, -7.2, -7.2, -7.2, 13.8],
        [-25, -33, -41, -44, -44]
    ]),
        r1[:, 3, :])
    np.testing.assert_allclose(np.array([
        [0, -3.2, -3.2, -19.2, -24.6],
        [0, 40, 66, 156, 156]
    ]),
        r1[:, 4, :])
    np.testing.assert_allclose(np.array([
        [0, -4, -4, -4, -16.6],
        [0, -5, -57, -64.5, -232.5]
    ]),
        r1[:, 5, :])

    w1_copy = w1.copy()

    np.testing.assert_allclose(r1[:, 0, :],
                               fruits.ISS(X_1, [w1_copy])[:, 0, :])


def test_theoretical_cases():
    X = np.random.random_sample((25, 1, 100))
    for i in range(X.shape[0]):
        X[i, 0, :] = (X[i, 0, :] - X[i].mean(axis=-1)) / X[i].std(axis=-1)

    result = fruits.ISS(X, fruits.words.SimpleWord("[1][1]"))

    np.testing.assert_allclose(np.ones((25,)) * -50,
                               result[:, 0, -1])


def test_word_generation():
    for n in range(1, 7):
        assert len(fruits.words.of_weight(n, dim=1)) \
            == 2**(n - 1)
    assert len(fruits.words.of_weight(4, dim=2)) == 82
