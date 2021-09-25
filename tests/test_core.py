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
    if not "ReLU" in fruits.core.letters.get_available():
        @fruits.core.letter(name="ReLU")
        def relu(X, i):
            return X[i, :] * (X[i, :]>0)

    # word: [relu(0)][relu(1)]
    relu_word = fruits.core.Word("relu collection")
    relu_word.multiply(fruits.core.ExtendedLetter("ReLU(1)"))
    relu_word.multiply(fruits.core.ExtendedLetter("ReLU(2)"))

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
    X = np.random.random_sample((10, 3, 50))
    word = fruits.core.SimpleWord("[12][2][33]")
    word.alpha = [0.5, -0.2]
    result = fruits.core.ISS(X, word)[:, 0, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += X[m, 0, i] * X[m, 1, i] \
                                     * X[m, 1, j] \
                                     * X[m, 2, k]**2 \
                                     * np.exp(-0.5*(j-i-1)) \
                                     * np.exp(0.2*(k-j-1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

    X = np.random.random_sample((10, 10, 50))
    word = fruits.core.SimpleWord("[(10)12345][9][23]")
    word.alpha = [-0.45, -3.14]
    result = fruits.core.ISS(X, word)[:, 0, -1]
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
                                     * np.exp(0.45*(j-i-1)) \
                                     * np.exp(3.14*(k-j-1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

    X = np.random.random_sample((10, 10, 50))
    word = fruits.core.Word("[ABS(3)][ABS(1)SIMPLE(10)][ABS(5)SIMPLE(10)]")
    word.alpha = [0.99, -2.71]
    result = fruits.core.ISS(X, word)[:, 0, -1]
    the_result = np.zeros((X.shape[0]))
    for m in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(k):
                for i in range(j):
                    the_result[m] += np.abs(X[m, 2, i]) \
                                     * np.abs(X[m, 0, j]) * X[m, 9, j] \
                                     * np.abs(X[m, 4, k]) * X[m, 9, k] \
                                     * np.exp(-0.99*(j-i-1)) \
                                     * np.exp(2.71*(k-j-1))

    np.testing.assert_allclose(the_result, result, rtol=1e-02)

def test_cache_plan():
    words = [
        fruits.core.SimpleWord("[1][11][3][11]"),
        fruits.core.SimpleWord("[11][13][11][1][3]"),
        fruits.core.SimpleWord("[1][13][1]"),
        fruits.core.SimpleWord("[11][13][111][13][11]"),
        fruits.core.SimpleWord("[3][11][111]"),
        fruits.core.SimpleWord("[1][11][2]",),
        fruits.core.SimpleWord("[11][2]"),
        fruits.core.SimpleWord("[11][13][111][13][2]"),
        fruits.core.SimpleWord("[3][11][1112][21]"),
    ]

    cache_plan = fruits.core.iss.CachePlan(words)

    assert cache_plan._plan == [4, 5, 2, 3, 3, 1, 1, 1, 2]

def test_extented_mode_simple_1():
    X = np.random.random_sample((10, 3, 100))

    word = fruits.core.SimpleWord("[11][21][331][22]")

    result_extended = fruits.core.ISS(X, word, mode="extended")

    word1 = fruits.core.SimpleWord("[11]")
    word2 = fruits.core.SimpleWord("[11][12]")
    word3 = fruits.core.SimpleWord("[11][12][133]")
    word4 = fruits.core.SimpleWord("[11][12][133][22]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.core.ISS(X, word1, mode="single")
    result_single[:, 1:2, :] = fruits.core.ISS(X, word2, mode="single")
    result_single[:, 2:3, :] = fruits.core.ISS(X, word3, mode="single")
    result_single[:, 3:4, :] = fruits.core.ISS(X, word4, mode="single")

    np.testing.assert_allclose(result_single, result_extended)

def test_extended_mode_simple_2():
    X = np.random.random_sample((10, 3, 100))

    word = fruits.core.SimpleWord("[1][11][111][1111]")

    result_extended = fruits.core.ISS(X, word, mode="extended")

    word1 = fruits.core.SimpleWord("[1]")
    word2 = fruits.core.SimpleWord("[1][11]")
    word3 = fruits.core.SimpleWord("[1][11][111]")
    word4 = fruits.core.SimpleWord("[1][11][111][1111]")

    result_single = np.zeros((10, 4, 100))

    result_single[:, 0:1, :] = fruits.core.ISS(X, word1, mode="single")
    result_single[:, 1:2, :] = fruits.core.ISS(X, word2, mode="single")
    result_single[:, 2:3, :] = fruits.core.ISS(X, word3, mode="single")
    result_single[:, 3:4, :] = fruits.core.ISS(X, word4, mode="single")

    np.testing.assert_allclose(result_single, result_extended)

def test_extended_mode_simple_3():
    X = np.random.random_sample((10, 3, 100))
    
    words = [
        fruits.core.SimpleWord("[1][11][3][11]"),
        fruits.core.SimpleWord("[11][13][11][1][3]"),
        fruits.core.SimpleWord("[1][13][1]"),
        fruits.core.SimpleWord("[11][13][111][13][11]"),
        fruits.core.SimpleWord("[3][11][111]"),
        fruits.core.SimpleWord("[1][11][2]",),
        fruits.core.SimpleWord("[11][2]"),
        fruits.core.SimpleWord("[11][13][111][13][2]"),
        fruits.core.SimpleWord("[3][11][1112][21]"),
    ]

    all_words = [
        fruits.core.SimpleWord("[1]"),
        fruits.core.SimpleWord("[1][11]"),
        fruits.core.SimpleWord("[1][11][3]"),
        fruits.core.SimpleWord("[1][11][3][11]"),
        fruits.core.SimpleWord("[11]"),
        fruits.core.SimpleWord("[11][13]"),
        fruits.core.SimpleWord("[11][13][11]"),
        fruits.core.SimpleWord("[11][13][11][1]"),
        fruits.core.SimpleWord("[11][13][11][1][3]"),
        fruits.core.SimpleWord("[1][13]"),
        fruits.core.SimpleWord("[1][13][1]"),
        fruits.core.SimpleWord("[11][13][111]"),
        fruits.core.SimpleWord("[11][13][111][13]"),
        fruits.core.SimpleWord("[11][13][111][13][11]"),
        fruits.core.SimpleWord("[3]"),
        fruits.core.SimpleWord("[3][11]"),
        fruits.core.SimpleWord("[3][11][111]"),
        fruits.core.SimpleWord("[1][11][2]",),
        fruits.core.SimpleWord("[11][2]"),
        fruits.core.SimpleWord("[11][13][111][13][2]"),
        fruits.core.SimpleWord("[3][11][1112]"),
        fruits.core.SimpleWord("[3][11][1112][21]"),
    ]

    result_extended = fruits.core.ISS(X, words, mode="extended")

    result_single = fruits.core.ISS(X, all_words, mode="single")

    np.testing.assert_allclose(result_single, result_extended)

def test_extended_mode_general_1():
    X = np.random.random_sample((10, 3, 100))

    if not "ReLU" in fruits.core.letters.get_available():
        @fruits.core.letter(name="ReLU")
        def relu(X, i):
            return X[i, :] * (X[i, :]>0)

    if not "EXP" in fruits.core.letters.get_available():
        @fruits.core.letter(name="EXP")
        def exp(X, i):
            return np.exp(-X[i, :]/100)

    el1 = fruits.core.ExtendedLetter("ReLU(0)ReLU(0)")
    el2 = fruits.core.ExtendedLetter("ReLU(1)EXP(0)")
    el3 = fruits.core.ExtendedLetter("EXP(1)EXP(2)")

    relu_word = fruits.core.Word()
    relu_word.multiply(el1)
    relu_word.multiply(el2)
    relu_word.multiply(el3)

    relu_word1 = fruits.core.Word()
    relu_word1.multiply(el1)

    relu_word2 = fruits.core.Word()
    relu_word2.multiply(el1)
    relu_word2.multiply(el2)

    relu_word3 = fruits.core.Word()
    relu_word3.multiply(el1)
    relu_word3.multiply(el2)
    relu_word3.multiply(el3)

    result_extended = fruits.core.ISS(X, relu_word, mode="extended")

    result_single = np.zeros((10, 3, 100))

    result_single[:, 0:1, :] = fruits.core.ISS(X, relu_word1, mode="single")
    result_single[:, 1:2, :] = fruits.core.ISS(X, relu_word2, mode="single")
    result_single[:, 2:3, :] = fruits.core.ISS(X, relu_word3, mode="single")

    np.testing.assert_allclose(result_single, result_extended)

def test_word_generation():
    for n in range(1, 7):
        assert len(fruits.core.generation.simplewords_by_weight(n, dim=1)) \
                == 2**(n-1)
    assert len(fruits.core.generation.simplewords_by_weight(4, dim=2)) == 82
