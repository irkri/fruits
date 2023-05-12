import numpy as np

import fruits

X_1 = np.array([
    [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.]],
    [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.]]
])


def test_increments():
    X_1_1 = fruits.preparation.INC().fit_transform(X_1)
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


def test_nrm():
    np.testing.assert_allclose(np.array([
        [[0., 4.8/9, 4/9, 1., 1/9], [1., 8/9, 7/9, 7/9, 0.]],
        [[5/8, 1., 2/8, 6/8, 0.], [3/7.5, 7/7.5, 4/7.5, 1., 0.]],
    ]), fruits.preparation.NRM().fit_transform(X_1))


    np.testing.assert_allclose(np.array([
        [[3/12, 7.8/12, 7/12, 1., 4/12], [9/12, 8/12, 7/12, 7/12, 0.]],
        [[13/16, 1., 10/16, 14/16, 8/16], [3/16, 7/16, 4/16, 7.5/16, 0.]],
    ]), fruits.preparation.NRM(scale_dim=True).fit_transform(X_1))


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
    X = np.random.random_sample((46, 2, 189))
    rin = fruits.preparation.RIN(2, adaptive_width=True)
    rin.fit(X)

    rin._kernel = np.array([[4., 1.], [4., 1.]])
    rin._ndim_per_kernel = np.array([1, 1], dtype=np.int32)
    rin._dims_per_kernel = np.array([0, 1], dtype=np.int32)

    np.testing.assert_allclose(
        np.array([
            [[-4., 4.8, 15.2, 1.8, -8.], [2., -1., -9., -4., -7.]],
            [[5., 3., -26., -28., -14.], [-5., 4., 17., 7.5, 8.5]]
        ]),
        rin.transform(X_1)
    )

    X = np.random.random_sample((46, 2, 189))
    rin = fruits.preparation.RIN(
        width=2,
        adaptive_width=False,
        out_dim=1,
    )
    rin.fit(X)

    assert rin._kernel.shape == (2, 2)
    rin._kernel = np.array([[4., 1.], [2., 3.]])
    rin._ndim_per_kernel = np.array([2], dtype=np.int32)
    rin._dims_per_kernel = np.array([0, 1], dtype=np.int32)

    np.testing.assert_allclose(
        np.array([
            [[0., 0, 8.2, -.2, -15.]],
            [[0., 0., -17., -14.5, -12.5]]
        ]),
        rin.transform(X_1)
    )


def test_jld():
    X = np.random.random_sample((46, 100, 189))
    result = fruits.preparation.JLD(25).fit_transform(X)

    assert result.shape == (46, 25, 189)


def test_ffn():
    ffn1 = fruits.preparation.FFN(d_hidden=3, center=False, relu_out=False)

    ffn1._weights1 = np.array([-1, -2, 1], dtype=np.float64)
    ffn1._biases = np.zeros(3, dtype=np.float64)
    ffn1._weights2 = np.array([1, -1, 2], dtype=np.float64)

    np.testing.assert_allclose(np.array([
        [[-4., 1.6, 0., 10., -3.], [4., 2., 0., 0., -7.]],
        [[10., 16., 4., 12., 0.], [-5., -1, -4., -.5, -8.]]
    ]), ffn1.transform(X_1))

    ffn2 = fruits.preparation.FFN(d_hidden=3, center=False, relu_out=True)

    ffn2._weights1 = np.array([-1, -2, 1], dtype=np.float64)
    ffn2._biases = np.zeros(3, dtype=np.float64)
    ffn2._weights2 = np.array([1, -1, 2], dtype=np.float64)

    np.testing.assert_allclose(np.array([
        [[0., 1.6, 0., 10., 0.], [4., 2., 0., 0., 0.]],
        [[10., 16., 4., 12., 0.], [0., 0., 0., 0., 0.]]
    ]), ffn2.transform(X_1))
