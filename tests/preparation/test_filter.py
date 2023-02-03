import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_window():
    X = np.array([
        [[1, 2, 4, 5, 6],
         [11, 22, 33, 44, 55]],

        [[10, 20, 30, 40, 50],
         [111, 222, 333, 444, 555]]
    ], dtype=float)

    w = fruits.preparation.WIN(0.0, 0.7)
    result = w.fit_transform(X)
    np.testing.assert_allclose(np.array([
        [[1, 2, 0, 0, 0], [11, 22, 0, 0, 0]],
        [[10, 20, 30, 0, 0], [111, 222, 333, 0, 0]]
    ]), result)

    w = fruits.preparation.WIN(0.7, 1.0)
    result = w.fit_transform(X)
    np.testing.assert_allclose(np.array([
        [[0, 2, 4, 5, 6], [0, 22, 33, 44, 55]],
        [[0, 0, 30, 40, 50], [0, 0, 333, 444, 555]]
    ]), result)


def test_dot():
    d = fruits.preparation.DOT(0.4)

    np.testing.assert_allclose(np.array([
        [[0, 0.8, 0, 5, 0], [0, 1, 0, 0, 0]],
        [[0, 8, 0, 6, 0], [0, -1, 0, -0.5, 0]]
    ]), d.fit_transform(X_1))

    d = fruits.preparation.DOT(0.9)

    np.testing.assert_allclose(np.array([
        [[0, 0, 0, 5, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 6, 0], [0, 0, 0, -0.5, 0]]
    ]), d.fit_transform(X_1))

    d = fruits.preparation.DOT(0.1)

    X_2 = np.arange(100)
    X_2 = X_2[np.newaxis, np.newaxis, :]

    X_2_result = np.zeros(X_2.shape)
    X_2_result[:, :, 9::10] = X_2[:, :, 9::10]

    np.testing.assert_allclose(X_2_result, d.fit_transform(X_2))
