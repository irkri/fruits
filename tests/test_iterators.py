import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_fast_slow_iss():
    # word [11122][122222][11]
    word1 = [[lambda X: X[0, :]**3, lambda X: X[1, :]**2],
             [lambda X: X[0, :], lambda X: X[1, :]**5],
             [lambda X: X[0, :]**2]]

    # word [22][112][2221]
    word2 = [[lambda X: X[1, :]**2],
             [lambda X: X[0, :]**2, lambda X: X[1, :]],
             [lambda X: X[1, :]**3, lambda X: X[0, :]]]

    it1 = fruits.iterators.SummationIterator("word 1")
    it1.append(*word1)
    it2 = fruits.iterators.SummationIterator("word 2")
    it2.append(*word2)

    sit1 = fruits.iterators.SimpleWord("[11122][122222][11]")
    sit2 = fruits.iterators.SimpleWord("[22][112][2221]")

    result_fast = fruits.core.ISS(X_1, [sit1, sit2])
    result_slow = fruits.core.ISS(X_1, [it1, it2])

    np.testing.assert_allclose(result_slow, result_fast)

def test_simpleword_iss():
    w1 = fruits.iterators.SimpleWord("[1]")
    w2 = fruits.iterators.SimpleWord("[2]")
    w3 = fruits.iterators.SimpleWord("[11]")
    w4 = fruits.iterators.SimpleWord("[12]")
    w5 = fruits.iterators.SimpleWord("[1][1]")
    w6 = fruits.iterators.SimpleWord("[1][2]")

    r1 = fruits.core.ISS(X_1, [w1, w2, w3, w4, w5, w6])

    np.testing.assert_allclose(np.array([
                                    [-4,-3.2,-3.2,1.8,-1.2],
                                    [5,13,15,21,21]
                               ]),
                               r1[:,0,:])
    np.testing.assert_allclose(np.array([
                                    [2,3,3,3,-4],
                                    [-5,-6,-10,-10.5,-18.5]
                               ]),
                               r1[:,1,:])
    np.testing.assert_allclose(np.array([
                                    [16,16.64,16.64,41.64,50.64],
                                    [25,89,93,129,129]
                               ]),
                               r1[:,2,:])
    np.testing.assert_allclose(np.array([
                                    [-8,-7.2,-7.2,-7.2,13.8],
                                    [-25,-33,-41,-44,-44]
                               ]),
                               r1[:,3,:])
    np.testing.assert_allclose(np.array([
                                    [16,13.44,13.44,22.44,26.04],
                                    [25,129,159,285,285]
                               ]),
                               r1[:,4,:])
    np.testing.assert_allclose(np.array([
                                    [-8,-11.2,-11.2,-11.2,-2.8],
                                    [-25,-38,-98,-108.5,-276.5]
                               ]),
                               r1[:,5,:])

def test_complex_words():
    # word: [relu(0)][relu(1)]
    relu_iterator = fruits.iterators.SummationIterator("relu collection")
    relu_iterator.append([lambda X: X[0, :] * (X[0, :]>0)], 
                         [lambda X: X[1, :] * (X[1, :]>0)])
    mix = [relu_iterator, fruits.iterators.SimpleWord("[111]")]

    mix_result = fruits.core.ISS(X_1, mix)

    np.testing.assert_allclose(np.array([
                                    [[0,0.8,0.8,0.8,0.8],
                                     [-64,-63.488,-63.488,61.512,34.512]],
                                    [[0,0,0,0,0],
                                     [125,637,645,861,861]]
                               ]),
                               mix_result)
