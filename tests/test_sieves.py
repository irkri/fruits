import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_ppv():
    const_ppv = fruits.features.PPV(quantile=0, constant=True)
    ppv_1 = const_ppv.fit_sieve(X_1[0])
    ppv_2 = fruits.features.PPV(quantile=0.5, constant=False,
                                sample_size=1).fit_sieve(X_1[1])

    np.testing.assert_allclose(np.array([3/5,4/5]), ppv_1)
    np.testing.assert_allclose(np.array([1,0]), ppv_2)

    ppv_1_copy = const_ppv.copy().fit_sieve(X_1[0])

    np.testing.assert_allclose(ppv_1, ppv_1_copy)

    ppvc_1 = fruits.features.PPVC(quantile=0, constant=True)

    np.testing.assert_allclose([0.4,0.4], ppvc_1.fit_sieve(X_1[0]))

def test_min_max():
    max_ = fruits.features.MAX().fit_sieve(X_1[0])
    min_ = fruits.features.MIN().fit_sieve(X_1[1])

    np.testing.assert_allclose(np.array([5,2]), max_)
    np.testing.assert_allclose(np.array([0,-8]), min_)

    max_cut_1 = fruits.features.MAX(cut=3).fit_sieve(X_1[0])
    max_cut_2 = fruits.features.MAX(cut=0.5).fit_sieve(X_1[0])

    np.testing.assert_allclose(np.array([0.8,2]), max_cut_1)
    np.testing.assert_allclose(np.array([0.8,2]), max_cut_2)

    min_cut_1 = fruits.features.MIN(cut=3).fit_sieve(X_1[0])
    min_cut_2 = fruits.features.MIN(cut=0.5).fit_sieve(X_1[1])

    np.testing.assert_allclose(np.array([-4,0]), min_cut_1)
    np.testing.assert_allclose(np.array([5,-5]), min_cut_2)

def test_end():
    end = fruits.features.END().fit_sieve(X_1[0])

    np.testing.assert_allclose(np.array([-3,-7]), end)

    end_cut = fruits.features.END(cut=0.2).fit_sieve(X_1[0])

    np.testing.assert_allclose(np.array([-4,0]), end_cut)
