import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_fast_slow_iss():
    @fruits.core.letter
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
    
    word1 = fruits.core.Word("Word 1")
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
    
    word2 = fruits.core.Word("Word 2")
    word2.multiply(el_2_1)
    word2.multiply(el_2_2)
    word2.multiply(el_2_3)

    sit1 = fruits.core.SimpleWord("[11122][122222][11]")
    sit2 = fruits.core.SimpleWord("[22][112][2221]")

    X = np.random.random_sample((100,2,100))

    result_fast = fruits.core.ISS(X, [sit1, sit2])
    result_slow = fruits.core.ISS(X, [word1, word2])

    np.testing.assert_allclose(result_slow, result_fast)

    word1_copy = word1.copy()
    word2_copy = word2.copy()
    result_slow_copy = fruits.core.ISS(X, [word1_copy, word2_copy])

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
                                    [0,-3.2,-3.2,-19.2,-24.6],
                                    [0,40,66,156,156]
                               ]),
                               r1[:,4,:])
    np.testing.assert_allclose(np.array([
                                    [0,-4,-4,-4,-16.6],
                                    [0,-5,-57,-64.5,-232.5]
                               ]),
                               r1[:,5,:])

    w1_copy = w1.copy()

    np.testing.assert_allclose(r1[:,0,:],
                               fruits.core.ISS(X_1, [w1_copy])[:,0,:])

def test_general_words():
    @fruits.core.letter(name="ReLU")
    def relu(X, i):
        return X[i, :] * (X[i, :]>0)

    # word: [relu(0)][relu(1)]
    relu_word = fruits.core.Word("relu collection")
    relu_word.multiply(fruits.core.ExtendedLetter((relu, 0)))
    relu_word.multiply(fruits.core.ExtendedLetter((relu, 1)))
                                   
    mix = [relu_word, fruits.core.SimpleWord("[111]")]

    mix_result = fruits.core.ISS(X_1, mix)

    np.testing.assert_allclose(np.array([
                                    [[0,0,0,0,0],
                                     [-64,-63.488,-63.488,61.512,34.512]],
                                    [[0,0,0,0,0],
                                     [125,637,645,861,861]]
                               ]),
                               mix_result)

def test_theoretical_cases():
    X = np.random.random_sample((25, 1, 100))
    for i in range(X.shape[0]):
        X[i, 0, :] = (X[i, 0, :] - X[i].mean(axis=-1)) / X[i].std(axis=-1)

    result = fruits.core.ISS(X, fruits.core.SimpleWord("[1][1]"))

    np.testing.assert_allclose(np.ones((25,))*-50,
                               result[:,0,-1])

def test_weighted_iss():
    X = np.random.random_sample((10, 3, 40))
    word = fruits.core.SimpleWord("[12][2][33]")
    word.alpha = [0.5, -0.2]
    result = fruits.core.ISS(X, word)[:, 0, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i]*X[m, 1, i] \
                                     * X[m, 1, j] \
                                     * X[m, 2, k]**2 \
                                     * np.exp(-0.5*(j-i)) \
                                     * np.exp(0.2*(k-j))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

def test_extented_mode():
    X = np.random.random_sample((10, 3, 100))

    word = fruits.core.SimpleWord("[11][21][331][22]")

    result_extended = fruits.core.ISS(X, word, mode="extended")

    word1 = fruits.core.SimpleWord("[11]")
    word2 = fruits.core.SimpleWord("[11][12]")
    word3 = fruits.core.SimpleWord("[11][12][133]")
    word4 = fruits.core.SimpleWord("[11][12][133][22]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.core.ISS(X, word1, mode="whole")
    result_single[:, 1:2, :] = fruits.core.ISS(X, word2, mode="whole")
    result_single[:, 2:3, :] = fruits.core.ISS(X, word3, mode="whole")
    result_single[:, 3:4, :] = fruits.core.ISS(X, word4, mode="whole")

    np.testing.assert_allclose(result_single, result_extended)

    word = fruits.core.SimpleWord("[1][11][111][1111]")

    result_extended = fruits.core.ISS(X, word, mode="extended")

    word1 = fruits.core.SimpleWord("[1]")
    word2 = fruits.core.SimpleWord("[1][11]")
    word3 = fruits.core.SimpleWord("[1][11][111]")
    word4 = fruits.core.SimpleWord("[1][11][111][1111]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.core.ISS(X, word1, mode="whole")
    result_single[:, 1:2, :] = fruits.core.ISS(X, word2, mode="whole")
    result_single[:, 2:3, :] = fruits.core.ISS(X, word3, mode="whole")
    result_single[:, 3:4, :] = fruits.core.ISS(X, word4, mode="whole")

    np.testing.assert_allclose(result_single, result_extended)

def test_word_generation():
    for n in range(1, 7):
        assert len(fruits.core.generation.simplewords_by_weight(n, dim=1)) \
                == 2**(n-1)
    assert len(fruits.core.generation.simplewords_by_weight(4, dim=2)) == 82

    assert len(fruits.core.generation.simplewords_by_degree(2, 2, 1)) == 6
    assert len(fruits.core.generation.simplewords_by_degree(2, 3, 1)) == 14
    assert len(fruits.core.generation.simplewords_by_degree(2, 2, 2)) == 30
