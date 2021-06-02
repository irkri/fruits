import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_increments():
    X_1_1 = fruits.preparateurs.INC(True).fit_prepare(X_1)
    increments = fruits.preparateurs.INC(zero_padding=False)
    X_1_2 = increments.fit_prepare(X_1)

    np.testing.assert_allclose(np.array([
                                [[0,4.8,-0.8,5,-8], [0,-1,-1,0,-7]],
                                [[0,3,-6,4,-6], [0,4,-3,3.5,-7.5]]
                               ]),
                               X_1_1)
    np.testing.assert_allclose(np.array([
                                [[-4,4.8,-0.8,5,-8], [2,-1,-1,0,-7]],
                                [[5,3,-6,4,-6], [-5,4,-3,3.5,-7.5]]
                               ]),
                               X_1_2)

    X_1_2_copy = increments.copy().fit_prepare(X_1)

    np.testing.assert_allclose(X_1_2, X_1_2_copy)

def test_standardization():
    X_1_1 = fruits.preparateurs.STD().fit_prepare(X_1)

    np.testing.assert_almost_equal(0, np.mean(X_1_1.flatten()))
    np.testing.assert_almost_equal(1, np.std(X_1_1.flatten()))
