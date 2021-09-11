import pytest
import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_ppv():
    ppv_1 = fruits.sieving.PPV(quantile=0, constant=True)
    ppv_2 = fruits.sieving.PPV(quantile=0.5, constant=False,
                                sample_size=1)

    np.testing.assert_allclose(np.array([[3/5],[4/5]]),
                               ppv_1.fit_sieve(X_1[0]))
    np.testing.assert_allclose(np.array([[1],[0]]),
                               ppv_2.fit_sieve(X_1[1]))

    ppv_1_copy = ppv_1.copy()

    np.testing.assert_allclose(ppv_1.fit_sieve(X_1[0]),
                               ppv_1_copy.fit_sieve(X_1[0]))

    ppvc_1 = fruits.sieving.PCC(quantile=0, constant=True)

    np.testing.assert_allclose([[0.4],[0.4]], ppvc_1.fit_sieve(X_1[0]))

    with pytest.raises(ValueError):
        fruits.sieving.PPV(quantile=[0.5,0.1,-1], constant=[False,True,False])

    with pytest.raises(ValueError):
        fruits.sieving.PCC(quantile=[0.5,2], constant=False)

    ppv_group_1 = fruits.sieving.PPV(quantile=[0.5,0.1,0.7],
                                     constant=False,
                                     sample_size=1,
                                     segments=False)

    np.testing.assert_allclose(np.array([[1,1,3/5],[0,4/5,0]]),
                               ppv_group_1.fit_sieve(X_1[1]))

    ppv_group_2 = fruits.sieving.PPV(quantile=[0.5,0.1,0.7],
                                     constant=False,
                                     sample_size=1,
                                     segments=True)

    np.testing.assert_allclose(np.array([[0,2/5],[4/5,0]]),
                               ppv_group_2.fit_sieve(X_1[1]))

    ppv_group_3 = fruits.sieving.PPV(quantile=[-5,0,2],
                                     constant=True,
                                     sample_size=1,
                                     segments=False)

    np.testing.assert_allclose(np.array([[1,1,4/5],[4/5,0,0]]),
                               ppv_group_3.fit_sieve(X_1[1]))

    ppv_group_4 = fruits.sieving.PPV(quantile=[0,-5,2],
                                     constant=True,
                                     sample_size=1,
                                     segments=True)

    np.testing.assert_allclose(np.array([[0,1/5],[4/5,0]]),
                               ppv_group_4.fit_sieve(X_1[1]))

    ppv_group_4_copy = ppv_group_4.copy()

    np.testing.assert_allclose(ppv_group_4.fit_sieve(X_1[1]),
                               ppv_group_4_copy.fit_sieve(X_1[1]))

def test_max():
    max_ = fruits.sieving.MAX()

    np.testing.assert_allclose(np.array([[5],[2]]), max_.fit_sieve(X_1[0]))

    max_cut_1 = fruits.sieving.MAX(cut=3)
    max_cut_2 = fruits.sieving.MAX(cut=0.5)

    np.testing.assert_allclose(np.array([[0.8],[2]]),
                               max_cut_1.fit_sieve(X_1[0]))
    np.testing.assert_allclose(np.array([[5],[2]]),
                               max_cut_2.fit_sieve(X_1[0]))

    max_cut_group_1 = fruits.sieving.MAX(cut=[-1, 3, 1],
                                         segments=False)

    np.testing.assert_allclose(np.array([[5,0.8,-4],[2,2,2]]),
                               max_cut_group_1.fit_sieve(X_1[0]))

    max_cut_group_2 = fruits.sieving.MAX(cut=[-1, 0.2, 0.7, 0.5],
                                         segments=True)

    np.testing.assert_allclose(np.array([[5,5,5],[0,0,0]]),
                               max_cut_group_2.fit_sieve(X_1[0]))

    max_cut_group_2_copy = max_cut_group_2.copy()

    np.testing.assert_allclose(max_cut_group_2.fit_sieve(X_1[0]),
                               max_cut_group_2_copy.fit_sieve(X_1[0]))

def test_min():
    min_ = fruits.sieving.MIN()

    np.testing.assert_allclose(np.array([[0],[-8]]), min_.fit_sieve(X_1[1]))

    min_cut_1 = fruits.sieving.MIN(cut=3)
    min_cut_2 = fruits.sieving.MIN(cut=0.5)

    np.testing.assert_allclose(np.array([[-4],[0]]),
                               min_cut_1.fit_sieve(X_1[0]))
    np.testing.assert_allclose(np.array([[2],[-5]]),
                               min_cut_2.fit_sieve(X_1[1]))

    min_cut_group_1 = fruits.sieving.MIN(cut=[-1, 3, 1],
                                         segments=False)

    np.testing.assert_allclose(np.array([[0,2,5],[-8,-5,-5]]),
                               min_cut_group_1.fit_sieve(X_1[1]))

    min_cut_group_2 = fruits.sieving.MIN(cut=[-1, 0.2, 0.7, 0.5],
                                         segments=True)

    np.testing.assert_allclose(np.array([[2,2,0],[-4,-0.5,-8]]),
                               min_cut_group_2.fit_sieve(X_1[1]))

    min_cut_group_2_copy = min_cut_group_2.copy()

    np.testing.assert_allclose(min_cut_group_2.fit_sieve(X_1[0]),
                               min_cut_group_2_copy.fit_sieve(X_1[0]))

def test_end():
    end = fruits.sieving.END()

    np.testing.assert_allclose(np.array([[-3],[-7]]), end.fit_sieve(X_1[0]))

    end_cut = fruits.sieving.END(cut=0.2)

    np.testing.assert_allclose(np.array([[-4],[0]]), end_cut.fit_sieve(X_1[0]))

    end_cut_group = fruits.sieving.END(cut=[1, 0.2, 0.8, 4, -1])

    np.testing.assert_allclose(np.array([[-4,-4,5,5,-3],[2,0,0,0,-7]]),
                               end_cut_group.fit_sieve(X_1[0]))

    end_cut_group_copy = end_cut_group.copy()

    np.testing.assert_allclose(end_cut_group.fit_sieve(X_1[0]),
                               end_cut_group_copy.fit_sieve(X_1[0]))

def test_pia():
    pia = fruits.sieving.PIA()

    np.testing.assert_allclose(np.array([[2/5],[1/5]]), pia.fit_sieve(X_1[0]))

    pia_cut_1 = fruits.sieving.PIA(cut=3, div_on_slice=True)
    pia_cut_2 = fruits.sieving.PIA(cut=0.5, div_on_slice=False)

    np.testing.assert_allclose(np.array([[1/3],[1/3]]),
                               pia_cut_1.fit_sieve(X_1[0]))
    np.testing.assert_allclose(np.array([[2/5],[2/5]]),
                               pia_cut_2.fit_sieve(X_1[1]))

    pia_cut_group_1 = fruits.sieving.PIA(cut=[-1, 3, 1],
                                         segments=False,
                                         div_on_slice=False)

    np.testing.assert_allclose(np.array([[3/5,2/5,1/5],[2/5,1/5,0]]),
                               pia_cut_group_1.fit_sieve(X_1[1]))

    pia_cut_group_2 = fruits.sieving.PIA(cut=[-1, 0.2, 0.7, 0.5],
                                         segments=True,
                                         div_on_slice=True)

    np.testing.assert_allclose(np.array([[1/2,1/2,1/2],[2/3,1/1,1/2]]),
                               pia_cut_group_2.fit_sieve(X_1[1]))

    pia_cut_group_2_copy = pia_cut_group_2.copy()

    np.testing.assert_allclose(pia_cut_group_2.fit_sieve(X_1[0]),
                               pia_cut_group_2_copy.fit_sieve(X_1[0]))

def test_lcs():
    lcs = fruits.sieving.LCS()

    np.testing.assert_allclose(np.array([[5],[5]]), lcs.fit_sieve(X_1[0]))

    lcs_cut_1 = fruits.sieving.LCS(cut=3)
    lcs_cut_2 = fruits.sieving.LCS(cut=0.5)

    np.testing.assert_allclose(np.array([[3],[3]]),
                               lcs_cut_1.fit_sieve(X_1[0]))
    np.testing.assert_allclose(np.array([[3],[4]]),
                               lcs_cut_2.fit_sieve(X_1[1]))

    lcs_cut_group_1 = fruits.sieving.LCS(cut=[-1, 3, 1],
                                         segments=False)

    np.testing.assert_allclose(np.array([[5,3,1],[5,3,1]]),
                               lcs_cut_group_1.fit_sieve(X_1[1]))

    lcs_cut_group_2 = fruits.sieving.LCS(cut=[-1, 0.2, 0.7, 0.5],
                                         segments=True)

    np.testing.assert_allclose(np.array([[2,2,2],[3,1,2]]),
                               lcs_cut_group_2.fit_sieve(X_1[1]))

    lcs_cut_group_2_copy = lcs_cut_group_2.copy()

    np.testing.assert_allclose(lcs_cut_group_2.fit_sieve(X_1[0]),
                               lcs_cut_group_2_copy.fit_sieve(X_1[0]))

def test_fruitstrings():
    sieve01 = fruits.sieving.MAX(cut=[1, 0.2, 0.4, 0.6, 0.8, -1],
                                 segments=False)
    sieve02 = fruits.sieving.MIN(cut=[1, 0.2, 0.4, 0.6, 0.8, -1],
                                 segments=True)
    sieve03 = fruits.sieving.PPV([0.1, 0.5], constant=False)
    sieve04 = fruits.sieving.PPV([0.1, 0.5], constant=True)
    sieve05 = fruits.sieving.MAX(cut=[5, 6, 20])

    assert sieve01._requisite == sieve02._requisite
    assert sieve03._requisite == sieve04._requisite
    assert sieve04._requisite == sieve05._requisite
    assert sieve01._requisite != sieve04._requisite
    assert sieve02._requisite != sieve03._requisite
