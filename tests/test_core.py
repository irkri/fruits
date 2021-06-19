import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_fast_slow_iss():
    @fruits.core.complex_letter
    def dim_letter(X, i):
        return X[i, :]

    # word [11122][122222][11]
    el_1_1 = fruits.core.ExtendedLetter()
    for i in range(3):
        el_1_1.append(dim_letter, 0)
    for i in range(2):
        el_1_1.append(dim_letter, 1)

    el_1_2 = fruits.core.ExtendedLetter()
    for i in range(1):
        el_1_2.append(dim_letter, 0)
    for i in range(5):
        el_1_2.append(dim_letter, 1)

    el_1_3 = fruits.core.ExtendedLetter()
    for i in range(2):
        el_1_3.append(dim_letter, 0)
    
    word1 = fruits.core.ComplexWord("Word 1")
    word1.multiply(el_1_1)
    word1.multiply(el_1_2)
    word1.multiply(el_1_3)

    # word [22][112][2221]
    el_2_1 = fruits.core.ExtendedLetter()
    for i in range(2):
        el_2_1.append(dim_letter, 1)

    el_2_2 = fruits.core.ExtendedLetter()
    for i in range(2):
        el_2_2.append(dim_letter, 0)
    for i in range(1):
        el_2_2.append(dim_letter, 1)

    el_2_3 = fruits.core.ExtendedLetter()
    for i in range(1):
        el_2_3.append(dim_letter, 0)
    for i in range(3):
        el_2_3.append(dim_letter, 1)
    
    word2 = fruits.core.ComplexWord("Word 2")
    word2.multiply(el_2_1)
    word2.multiply(el_2_2)
    word2.multiply(el_2_3)

    sit1 = fruits.core.SimpleWord("[11122][122222][11]")
    sit2 = fruits.core.SimpleWord("[22][112][2221]")

    result_fast = fruits.core.ISS(X_1, [sit1, sit2])
    result_slow = fruits.core.ISS(X_1, [word1, word2])

    np.testing.assert_allclose(result_slow, result_fast)

    word1_copy = word1.copy()
    word2_copy = word2.copy()
    result_slow_copy = fruits.core.ISS(X_1, [word1_copy, word2_copy])

    np.testing.assert_allclose(result_slow, result_slow_copy)

def test_simpleword_iss():
    w1 = fruits.core.SimpleWord("[1]")
    w2 = fruits.core.SimpleWord("[2]")
    w3 = fruits.core.SimpleWord("[11]")
    w4 = fruits.core.SimpleWord("[12]")
    w5 = fruits.core.SimpleWord("[1][1]")
    w6 = fruits.core.SimpleWord("[1][2]")

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

    w1_copy = w1.copy()

    np.testing.assert_allclose(r1[:,0,:],
                               fruits.core.ISS(X_1, [w1_copy])[:,0,:])

def test_complex_words():
    @fruits.core.complex_letter(name="ReLU")
    def relu(X, i):
        return X[i, :] * (X[i, :]>0)

    # word: [relu(0)][relu(1)]
    relu_iterator = fruits.core.ComplexWord("relu collection")
    relu_iterator.multiply(fruits.core.ExtendedLetter((relu, 0)))
    relu_iterator.multiply(fruits.core.ExtendedLetter((relu, 1)))
                                   
    mix = [relu_iterator, fruits.core.SimpleWord("[111]")]

    mix_result = fruits.core.ISS(X_1, mix)

    np.testing.assert_allclose(np.array([
                                    [[0,0.8,0.8,0.8,0.8],
                                     [-64,-63.488,-63.488,61.512,34.512]],
                                    [[0,0,0,0,0],
                                     [125,637,645,861,861]]
                               ]),
                               mix_result)

def test_word_generation():
    for n in range(1, 7):
        assert len(fruits.core.generation.simplewords_by_length(n, dim=1)) \
                == 2**(n-1)
    assert len(fruits.core.generation.simplewords_by_length(4, dim=2)) == 82

    assert len(fruits.core.generation.simplewords_by_degree(2, 2, 1)) == 6
    assert len(fruits.core.generation.simplewords_by_degree(2, 3, 1)) == 14
    assert len(fruits.core.generation.simplewords_by_degree(2, 2, 2)) == 30
