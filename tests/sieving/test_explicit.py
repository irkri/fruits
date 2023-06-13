import numpy as np

import fruits

X_1 = np.array([
    [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.]],
    [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.]]
])


def test_max():
    max_ = fruits.sieving.MAX()

    np.testing.assert_allclose(np.array([[5], [2]]),
                               max_.fit_transform(X_1[0]))

    max_cut_1 = fruits.sieving.MAX(cut=3)
    max_cut_2 = fruits.sieving.MAX(cut=0.5)

    np.testing.assert_allclose(np.array([[0.8], [2]]),
                               max_cut_1.fit_transform(X_1[0]))
    np.testing.assert_allclose(np.array([[5], [2]]),
                               max_cut_2.fit_transform(X_1[0]))

    max_cut_group_1 = fruits.sieving.MAX(cut=[-1, 3, 1])

    np.testing.assert_allclose(np.array([[-4, 0.8, 5], [2, 1, 0]]),
                               max_cut_group_1.fit_transform(X_1[0]))

    max_cut_group_2 = fruits.sieving.MAX(cut=[-1, 0.2, 0.7, 0.5])

    np.testing.assert_allclose(np.array([[-4, 5, 0, -3], [2, 0, 0, -7]]),
                               max_cut_group_2.fit_transform(X_1[0]))

    max_cut_group_2_copy = max_cut_group_2.copy()

    np.testing.assert_allclose(max_cut_group_2.fit_transform(X_1[0]),
                               max_cut_group_2_copy.fit_transform(X_1[0]))


def test_min():
    min_ = fruits.sieving.MIN()

    np.testing.assert_allclose(np.array([[0], [-8]]),
                               min_.fit_transform(X_1[1]))

    min_cut_1 = fruits.sieving.MIN(cut=3)
    min_cut_2 = fruits.sieving.MIN(cut=0.5)

    np.testing.assert_allclose(np.array([[-4], [0]]),
                               min_cut_1.fit_transform(X_1[0]))
    np.testing.assert_allclose(np.array([[2], [-5]]),
                               min_cut_2.fit_transform(X_1[1]))

    min_cut_group_1 = fruits.sieving.MIN(cut=[-1, 3, 1])

    np.testing.assert_allclose(np.array([[5, 2, 0], [-5, -4, -8]]),
                               min_cut_group_1.fit_transform(X_1[1]))

    min_cut_group_2 = fruits.sieving.MIN(cut=[-1, 0.2, 0.7, 0.5])

    np.testing.assert_allclose(np.array([[5, 2, 6, 0], [-5, -4, 0, -8]]),
                               min_cut_group_2.fit_transform(X_1[1]))

    min_cut_group_2_copy = min_cut_group_2.copy()

    np.testing.assert_allclose(min_cut_group_2.fit_transform(X_1[0]),
                               min_cut_group_2_copy.fit_transform(X_1[0]))


def test_end():
    end = fruits.sieving.END()

    np.testing.assert_allclose(np.array([[-3], [-7]]),
                               end.fit_transform(X_1[0]))

    end_cut = fruits.sieving.END(cut=0.2)

    np.testing.assert_allclose(
        np.array([[-4], [0]]), end_cut.fit_transform(X_1[0]))

    end_cut_group = fruits.sieving.END(cut=[1, 0.2, 0.8, 4, -1])

    np.testing.assert_allclose(
        np.array([[-4, -4, 5, 5, -3], [2, 0, 0, 0, -7]]),
        end_cut_group.fit_transform(X_1[0])
    )

    end_cut_group_copy = end_cut_group.copy()

    np.testing.assert_allclose(end_cut_group.fit_transform(X_1[0]),
                               end_cut_group_copy.fit_transform(X_1[0]))


def test_npi():
    npi = fruits.sieving.NPI()

    np.testing.assert_allclose(
        np.array([[2], [0]]), npi.fit_transform(X_1[0]))

    npi_cut_1 = fruits.sieving.NPI(cut=3)
    npi_cut_2 = fruits.sieving.NPI(cut=0.5)

    np.testing.assert_allclose(
        np.array([[1], [0]]),
        npi_cut_1.fit_transform(X_1[0]),
    )
    np.testing.assert_allclose(
        np.array([[1], [2]]),
        npi_cut_2.fit_transform(X_1[1]),
    )

    npi_cut_group_1 = fruits.sieving.NPI(cut=[-1, 3, 1])

    np.testing.assert_allclose(
        np.array([[0, 1, 1], [0, 1, 1]]),
        npi_cut_group_1.fit_transform(X_1[1]),
    )

    npi_cut_group_2 = fruits.sieving.NPI(cut=[-1, 0.2, 0.7, 0.5])

    cache = fruits.cache.SharedSeedCache()
    for key in ["0.2", "0.7", "0.5"]:
        cache.get(
            fruits.cache.CacheType.COQUANTILE,
            key+":L2",
            np.array([[[5, 8, 2, 6, 0]], [[-5, -1, -4, -0.5, -8]]]),
        )

    np.testing.assert_allclose(np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0]]
    ), npi_cut_group_2.fit_transform(X_1[1]))

    npi_cut_group_2_copy = npi_cut_group_2.copy()

    np.testing.assert_allclose(npi_cut_group_2.fit_transform(X_1[0]),
                               npi_cut_group_2_copy.fit_transform(X_1[0]))


def test_mpi():
    mpi = fruits.sieving.MPI()

    np.testing.assert_allclose(
        np.array([[4.9], [0]]), mpi.fit_transform(X_1[0]),
    )
    np.testing.assert_allclose(
        np.array([[3.5], [3.75]]), mpi.fit_transform(X_1[1]),
    )


def test_xpi():
    xpi = fruits.sieving.XPI()

    np.testing.assert_allclose(
        np.array([[2], [0]]), xpi.fit_transform(X_1[0]),
    )
    np.testing.assert_allclose(
        np.array([[2], [2]]), xpi.fit_transform(X_1[1]),
    )


def test_lpi():
    lpi = fruits.sieving.LPI()

    np.testing.assert_allclose(
        np.array([[1], [0]]), lpi.fit_transform(X_1[0]),
    )
    np.testing.assert_allclose(
        np.array([[1], [1]]), lpi.fit_transform(X_1[1]),
    )
