import numpy as np

import fruits


def test_index_weighting():
    X = np.random.random_sample((10, 3, 50))
    word = fruits.words.SimpleWord("[12][2][33]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.weighting.Indices(scalars=[.6, .2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.exp(.6 * (i/X.shape[2] - j/X.shape[2])) \
                        * np.exp(.2 * (j/X.shape[2] - k/X.shape[2]))

    np.testing.assert_allclose(the_result, result, rtol=1e-05)

    X = np.random.random_sample((10, 10, 50))
    word = fruits.words.SimpleWord("[(10)12345][9][23]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.weighting.Indices(scalars=[.45, 3.14]),
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
                        * np.exp(.45 * (i/X.shape[2] - j/X.shape[2])) \
                        * np.exp(3.14 * (j/X.shape[2] - k/X.shape[2]))

    np.testing.assert_allclose(the_result, result, rtol=1e-05)


def test_L1_weighting():
    X = np.random.random_sample((10, 3, 50))
    Y = np.zeros_like(X)
    Y[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]
    Y = np.cumsum(np.abs(Y[:, 0, :]), axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= Y[i, -1]
    word = fruits.words.SimpleWord("[12][2][33]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.weighting.L1([.6, .2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.exp(.6 * (Y[m, i] - Y[m, j])) \
                        * np.exp(.2 * (Y[m, j] - Y[m, k]))

    np.testing.assert_allclose(the_result, result, rtol=1e-06)


def test_L2_weighting():
    X = np.random.random_sample((10, 3, 50))
    Y = np.zeros_like(X)
    Y[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]
    Y = np.cumsum(Y[:, 0, :]**2, axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= Y[i, -1]
    word = fruits.words.SimpleWord("[12][2][33]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.weighting.L2([.6, .2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 1, j] \
                        * X[m, 2, k]**2 \
                        * np.exp(.6 * (Y[m, i] - Y[m, j])) \
                        * np.exp(.2 * (Y[m, j] - Y[m, k]))

    np.testing.assert_allclose(the_result, result, rtol=1e-06)

    X = np.random.random_sample((10, 3, 50))
    Y = np.zeros_like(X)
    Y[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]
    Y = np.cumsum(Y[:, 0, :]**2, axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= Y[i, -1]
    word = fruits.words.SimpleWord("[12][3][2213]")
    result = fruits.ISS(
        [word],
        weighting=fruits.iss.weighting.L2(),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                        * X[m, 2, j] \
                        * X[m, 1, k]**2 * X[m, 0, k] * X[m, 2, k] \
                        * np.exp(Y[m, i] - Y[m, k])

    np.testing.assert_allclose(the_result, result, rtol=1e-06)


def test_arctic_weighting():
    X = np.random.random_sample((10, 3, 50))
    Y = np.zeros_like(X)
    Y[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]
    Y = np.cumsum(np.abs(Y[:, 0, :]), axis=1)
    word = fruits.words.SimpleWord("[12][-3][2213]")
    result = fruits.ISS(
        [word],
        semiring=fruits.semiring.Arctic(),
        weighting=fruits.iss.weighting.L1([.5, .2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0])) - np.inf
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k+1):
                for i in range(j+1):
                    the_result[m] = max(the_result[m],
                        X[m, 0, i] + X[m, 1, i]
                        - X[m, 2, j]
                        + 2*X[m, 1, k] + X[m, 0, k] + X[m, 2, k]
                        + .2 * (Y[m, j] - Y[m, k]) + .5 * (Y[m, i] - Y[m, j])
                    )

    np.testing.assert_allclose(the_result, result, rtol=1e-06)


def test_bayesian_weighting():
    X = np.random.random_sample((10, 3, 50))
    Y = np.zeros_like(X)
    Y[:, :, 1:] = X[:, :, 1:] - X[:, :, :-1]
    Y = np.cumsum(Y[:, 0, :]**2, axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= Y[i, -1]
    word = fruits.words.SimpleWord("[12][3][2213]")
    result = fruits.ISS(
        [word],
        semiring=fruits.semiring.Bayesian(),
        weighting=fruits.iss.weighting.L2([.6, .2]),
    ).fit_transform(X)[0, :, -1]
    the_result = np.zeros((X.shape[0])) - np.inf
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k+1):
                for i in range(j+1):
                    the_result[m] = max(the_result[m],
                        X[m, 0, i] * X[m, 1, i]
                        * X[m, 2, j]
                        * X[m, 1, k]**2 * X[m, 0, k] * X[m, 2, k]
                        * np.exp(.6 * (Y[m, i] - Y[m, j])) \
                        * np.exp(.2 * (Y[m, j] - Y[m, k]))
                    )

    np.testing.assert_allclose(the_result, result, rtol=1e-06)
