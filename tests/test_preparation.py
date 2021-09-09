import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_increments():
    X_1_1 = fruits.preparation.INC(True).fit_prepare(X_1)
    increments = fruits.preparation.INC(zero_padding=False)
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
    X_1_1 = fruits.preparation.STD().fit_prepare(X_1)

    np.testing.assert_almost_equal(0, np.mean(X_1_1.flatten()))
    np.testing.assert_almost_equal(1, np.std(X_1_1.flatten()))

def test_window():
    X = np.array([
                  [[1,2,4,5,6],
                   [11,22,33,44,55]],

                  [[10,20,30,40,50],
                   [111,222,333,444,555]]
                ])

    w = fruits.preparation.WIN(0.0, 0.7)
    result = w.fit_prepare(X)
    np.testing.assert_allclose(np.array([
                                [[1,2,0,0,0], [11,22,0,0,0]],
                                [[10,20,30,0,0], [111,222,333,0,0]]
                               ]),
                               result)

    w = fruits.preparation.WIN(0.7, 1.0)
    result = w.fit_prepare(X)
    np.testing.assert_allclose(np.array([
                                [[0,0,4,5,6], [0,0,33,44,55]],
                                [[0,0,0,40,50], [0,0,0,444,555]]
                               ]),
                               result)

    abs1 = fruits.core.wording.Word()
    el = fruits.core.letters.ExtendedLetter()
    el.append(fruits.core.letters.absolute)
    abs1.multiply(el)

    requisite = fruits.requisites.Requisite("INC -> ABS")
    requisite.configure(preparateur=fruits.preparation.INC(zero_padding=False),
                        word=abs1)
    fruits.requisites.log(requisite)

    w = fruits.preparation.WIN(0.0, 0.7)
    w.set_requisite("INC -> ABS")
    result = w.fit_prepare(X)
    np.testing.assert_allclose(np.array([
                                [[1,2,4,0,0], [11,22,33,0,0]],
                                [[10,20,30,0,0], [111,222,333,0,0]]
                               ]),
                               result)

def test_dot():
    d = fruits.preparation.DOT(0.4)

    np.testing.assert_allclose(np.array([
                                [[0,0.8,0,5,0], [0,1,0,0,0]],
                                [[0,8,0,6,0], [0,-1,0,-0.5,0]]
                               ]),
                               d.fit_prepare(X_1))

    d = fruits.preparation.DOT(0.9)

    np.testing.assert_allclose(np.array([
                                [[0,0,0,5,0], [0,0,0,0,0]],
                                [[0,0,0,6,0], [0,0,0,-0.5,0]]
                               ]),
                               d.fit_prepare(X_1))

    d = fruits.preparation.DOT(0.1)

    X_2 = np.arange(100)
    X_2 = X_2[np.newaxis, np.newaxis, :]

    X_2_result = np.zeros(X_2.shape)
    X_2_result[:, :, 9::10] = X_2[:, :, 9::10]

    np.testing.assert_allclose(X_2_result, d.fit_prepare(X_2))

def test_lag():
    L = fruits.preparation.LAG(1)

    np.testing.assert_allclose(np.array([
                                [[-4,0.8,0,5,-3], [0,-4,0.8,0,5],
                                 [2,1,0,0,-7], [0,2,1,0,0]],
                                [[5,8,2,6,0], [0,5,8,2,6],
                                 [-5,-1,-4,-0.5,-8], [0,-5,-1,-4,-0.5]]
                               ]),
                               L.fit_prepare(X_1))
