import numpy as np

import fruits

X_1 = np.array([
    [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.]],
    [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.]]
])


def test_lag():
    L = fruits.preparation.LAG()

    np.testing.assert_allclose(np.array([
        [
            [-4., 0.8, 0.8, 0., 0., 5., 5., -3., -3.],
            [-4., -4., 0.8, 0.8, 0., 0., 5., 5., -3.],
            [2., 1., 1., 0., 0., 0., 0., -7., -7.],
            [2., 2., 1., 1., 0., 0., 0., 0., -7.],
        ],
        [
            [5., 8., 8., 2., 2., 6., 6., 0., 0.],
            [5., 5., 8., 8., 2., 2., 6., 6., 0.],
            [-5., -1., -1., -4., -4., -0.5, -0.5, -8., -8.],
            [-5., -5., -1., -1., -4., -4., -0.5, -0.5, -8.],
        ],
    ]), L.fit_transform(X_1))


def test_one():
    one = fruits.preparation.ONE()

    np.testing.assert_allclose(np.array([
        [[-4., 0.8, 0., 5., -3.], [2., 1., 0., 0., -7.], [1., 1., 1., 1., 1.]],
        [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.], [1., 1., 1., 1., 1.]]
    ]), one.fit_transform(X_1))


def test_lay():
    lay = fruits.preparation.LAY(n=3, dim=0, std=0, overwrite=False)

    lay._weights1 = np.array([[-1, -2, 1]])
    lay._biases1 = np.array([6, -5, 0])
    lay._weights2 = np.array([1, -1, 2])

    np.testing.assert_allclose(np.array([
        [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.],
         [7., 6.8, 6., 11., 8.]],
        [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.],
         [11., 16., 8., 12., 6.]]
    ]), lay.transform(X_1))
