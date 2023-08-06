import numpy as np

import fruits


def test_exp1_weighting():
    X = np.random.random_sample((10, 3, 50))
    result = fruits.CosWISS([
            fruits.words.SimpleWord("[1][23]"),
            fruits.words.SimpleWord("[12][2][33]"),
        ],
        [0.5],
        exponent=1,
    ).fit_transform(X)[:, :, -1]
    the_result = np.zeros((2, X.shape[0]))
    g = np.pi * np.arange(X.shape[2]) / (0.5*(X.shape[2]-1))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                the_result[0, m] += X[m, 0, j] \
                    * X[m, 1, k] * X[m, 2, k] \
                    * np.cos(g[k] - g[j])
                for i in range(j):
                    the_result[1, m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.cos(g[j] - g[i]) \
                        * np.cos(g[k] - g[j])

    np.testing.assert_allclose(the_result, result, rtol=1e-05)

    X = np.random.random_sample((10, 3, 50))
    result = fruits.CosWISS([
            fruits.words.SimpleWord("[1][23]"),
            fruits.words.SimpleWord("[12][2][33]"),
        ],
        [0.5],
        exponent=1,
        total_weighting=True,
    ).fit_transform(X)[:, :, -1]
    the_result = np.zeros((2, X.shape[0]))
    g = np.pi * np.arange(X.shape[2]) / (0.5*(X.shape[2]-1))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                the_result[0, m] += X[m, 0, j] \
                    * X[m, 1, k] * X[m, 2, k] \
                    * np.cos(g[k] - g[j]) \
                    * np.cos(g[X.shape[2]-1] - g[k])
                for i in range(j):
                    the_result[1, m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.cos(g[j] - g[i]) \
                        * np.cos(g[k] - g[j]) \
                        * np.cos(g[X.shape[2]-1] - g[k])

    np.testing.assert_allclose(the_result, result, rtol=1e-05)


def test_exp2_weighting():
    X = np.random.random_sample((10, 3, 50))
    result = fruits.CosWISS([
            fruits.words.SimpleWord("[3][11]"),
            fruits.words.SimpleWord("[11][23][1]"),
        ],
        [0.5],
        exponent=2,
    ).fit_transform(X)[:, :, -1]
    the_result = np.zeros((2, X.shape[0]))
    g = np.pi * np.arange(X.shape[2]) / (0.5*(X.shape[2]-1))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                the_result[0, m] += X[m, 2, j] \
                    * X[m, 0, k]**2 \
                    * np.cos(g[k] - g[j])**2
                for i in range(j):
                    the_result[1, m] += X[m, 0, i]**2 \
                        * X[m, 1, j] * X[m, 2, j] \
                        * X[m, 0, k] \
                        * np.cos(g[j] - g[i])**2 \
                        * np.cos(g[k] - g[j])**2

    np.testing.assert_allclose(the_result, result, rtol=1e-05)


def test_exp3_weighting():
    X = np.random.random_sample((10, 1, 50))
    result = fruits.CosWISS([
            fruits.words.SimpleWord("[1][1][1][1]"),
        ],
        [0.5],
        exponent=3,
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros(X.shape[0])
    g = np.pi * np.arange(X.shape[2]) / (0.5*(X.shape[2]-1))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    for h in range(i):
                        the_result[m] += X[m, 0, i] \
                            * X[m, 0, j] \
                            * X[m, 0, k] \
                            * X[m, 0, h] \
                            * np.cos(g[i] - g[h])**3 \
                            * np.cos(g[j] - g[i])**3 \
                            * np.cos(g[k] - g[j])**3

    np.testing.assert_allclose(the_result, result, rtol=1e-05)


def test_exp4_weighting():
    X = np.random.random_sample((10, 1, 50))
    result = fruits.CosWISS([
            fruits.words.SimpleWord("[1][1][1]"),
        ],
        [0.5],
        exponent=4,
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros(X.shape[0])
    g = np.pi * np.arange(X.shape[2]) / (0.5*(X.shape[2]-1))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] \
                        * X[m, 0, j] \
                        * X[m, 0, k] \
                        * np.cos(g[j] - g[i])**4 \
                        * np.cos(g[k] - g[j])**4

    np.testing.assert_allclose(the_result, result, rtol=1e-05)
