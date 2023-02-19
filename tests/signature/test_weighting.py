import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2.0, 1, 0, 0, -7]],
    [[5.0, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_weighted_iss():
    X = np.random.random_sample((10, 3, 50))
    word = fruits.words.SimpleWord("[12][2][33]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.ExponentialWeighting([.5, -.2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.exp(-0.5 * (j - i - 1)) \
                        * np.exp(0.2 * (k - j - 1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

    X = np.random.random_sample((10, 10, 50))
    word = fruits.words.SimpleWord("[(10)12345][9][23]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.ExponentialWeighting([-.45, -3.14]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 9, i] * X[m, 0, i] \
                        * X[m, 1, i] * X[m, 2, i] \
                        * X[m, 3, i] * X[m, 4, i] \
                        * X[m, 8, j] \
                        * X[m, 1, k] * X[m, 2, k] \
                                     * np.exp(0.45 * (j - i - 1)) \
                                     * np.exp(3.14 * (k - j - 1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

    X = np.random.random_sample((10, 10, 50))
    word = fruits.words.Word("[ABS(3)][ABS(1)DIM(10)][ABS(5)DIM(10)]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.ExponentialWeighting([.99, -2.71]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += np.abs(X[m, 2, i]) \
                        * np.abs(X[m, 0, j]) * X[m, 9, j] \
                        * np.abs(X[m, 4, k]) * X[m, 9, k] \
                        * np.exp(-0.99 * (j - i - 1)) \
                        * np.exp(2.71 * (k - j - 1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)
