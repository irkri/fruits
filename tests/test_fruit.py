import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_n_features():
    featex = fruits.Fruit()

    featex.add(fruits.preparateurs.INC(zero_padding=False))

    featex.add(fruits.iterators.generate_words(1,3,5))
    featex.add(fruits.iterators.generate_random_words(12,2,3,5))

    assert len(featex.get_summation_iterators()) == 375

    featex.add(fruits.features.PPV(quantile=0, constant=True))
    featex.add(fruits.features.PPV(quantile=0.2, constant=False, 
                                   sample_size=1))
    featex.add(fruits.features.MAX)
    featex.add(fruits.features.MIN)

    assert featex.nfeatures() == 1500

    featex_copy = featex.copy()

    assert featex_copy.nfeatures() == 1500

    del featex

def test_branches():
    featex = fruits.Fruit()

    w1 = fruits.iterators.SimpleWord("[1]")
    w2 = fruits.iterators.SimpleWord("[2]")
    w3 = fruits.iterators.SimpleWord("[11]")
    w4 = fruits.iterators.SimpleWord("[12]")
    w5 = fruits.iterators.SimpleWord("[1][1]")
    w6 = fruits.iterators.SimpleWord("[1][2]")

    featex.add(w1, w2, w3)
    featex.add(fruits.features.MAX)
    featex.start_new_branch()
    featex.add(w4, w5, w6)
    featex.add(fruits.features.MIN)

    assert featex.nfeatures() == 6

    features = featex(X_1)

    assert features.shape == (2, 6)

    np.testing.assert_allclose(np.array([
                                    [1.8,3,50.64,-8,13.44,-11.2],
                                    [21,-5,129,-44,25,-276.5]
                               ]),
                               features)
