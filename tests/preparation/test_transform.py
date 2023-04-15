import numpy as np

import fruits

X_1 = np.array([
    [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.]],
    [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.]]
])


def test_increments():
    X_1_1 = fruits.preparation.INC(True).fit_transform(X_1)
    increments = fruits.preparation.INC(zero_padding=False)
    X_1_2 = increments.fit_transform(X_1)

    np.testing.assert_allclose(np.array([
        [[0., 4.8, -0.8, 5., -8.], [0., -1., -1., 0., -7.]],
        [[0., 3., -6., 4., -6.], [0., 4., -3., 3.5, -7.5]]
    ]), X_1_1)
    np.testing.assert_allclose(np.array([
        [[-4., 4.8, -0.8, 5., -8.], [2., -1., -1., 0., -7.]],
        [[5., 3., -6., 4., -6.], [-5., 4., -3., 3.5, -7.5]]
    ]), X_1_2)

    X_1_2_copy = increments.copy().fit_transform(X_1)

    np.testing.assert_allclose(X_1_2, X_1_2_copy)


def test_standardization():
    X_1_1 = fruits.preparation.STD().fit_transform(X_1)

    np.testing.assert_almost_equal(0, np.mean(X_1_1.flatten()))
    np.testing.assert_almost_equal(1, np.std(X_1_1.flatten()))


def test_mav():
    result = fruits.preparation.MAV(2).fit_transform(X_1)

    np.testing.assert_allclose(np.array([
        [[0, -1.6, 0.4, 2.5, 1], [0, 1.5, 0.5, 0, -3.5]],
        [[0, 6.5, 5, 4, 3], [0, -3, -2.5, -2.25, -4.25]]
    ]), result)

    result = fruits.preparation.MAV(0.6).fit_transform(X_1)

    np.testing.assert_allclose(np.array([
        [[0, 0, -3.2, 5.8, 2.], [0, 0, 3., 1., -7.]],
        [[0, 0, 15., 16., 8.], [0, 0, -10., -5.5, -12.5]]
    ]) / 3, result)


def test_lag():
    result = fruits.preparation.LAG().fit_transform(X_1)

    np.testing.assert_allclose(np.array([
        [[-4., 0.8, 0.8, 0., 0., 5., 5., -3., -3.],
         [-4., -4., 0.8, 0.8, 0., 0., 5., 5., -3.],
         [2., 1., 1., 0., 0., 0., 0., -7., -7.],
         [2., 2., 1., 1., 0., 0., 0., 0., -7.]],
        [[5., 8., 8., 2., 2., 6., 6., 0., 0.],
         [5., 5., 8., 8., 2., 2., 6., 6., 0.],
         [-5., -1., -1., -4., -4., -0.5, -0.5, -8., -8.],
         [-5., -5., -1., -1., -4., -4., -0.5, -0.5, -8.]]
    ]), result)


def test_rin():
    X = np.random.random_sample((46, 100, 189))
    rin = fruits.preparation.RIN(2)
    rin.fit(X)

    rin._kernel = np.array([4., 1.])

    np.testing.assert_allclose(
        np.array([
            [[-4., 4.8, 15.2, 1.8, -8.], [2., -1., -9., -4., -7.]],
            [[5., 3., -26., -28., -14.], [-5., 4., 17., 7.5, 8.5]]
        ]),
        rin.transform(X_1)
    )


def test_jld():
    X = np.random.random_sample((46, 100, 189))
    result = fruits.preparation.JLD(25).fit_transform(X)

    assert result.shape == (46, 25, 189)
