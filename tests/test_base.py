import numpy as np

import fruits

X_1 = np.array([
                [[-4,0.8,0,5,-3], [2,1,0,0,-7]],
                [[5,8,2,6,0], [-5,-1,-4,-0.5,-8]]
               ])

def test_n_features():
    featex = fruits.Fruit()

    featex.add(fruits.preparation.INC(zero_padding=False))

    featex.add(fruits.core.generation.simplewords_by_degree(3, 5, 1))

    assert len(featex.branch().get_words()) == 363

    featex.add(fruits.sieving.PPV(quantile=0, constant=True))
    featex.add(fruits.sieving.PPV(quantile=0.2, constant=False, 
                                   sample_size=1))
    featex.add(fruits.sieving.PPV(quantile=[0.2,5], constant=[False,True], 
                                   sample_size=1, segments=True))
    featex.add(fruits.sieving.MAX(cut=[0.1,0.5,0.9], segments=True))
    featex.add(fruits.sieving.MIN(cut=[0.1,0.5,0.9], segments=False))

    assert featex.nfeatures() == 2904

    featex_copy = featex.deepcopy()

    assert featex_copy.nfeatures() == 2904

    del featex

def test_mode_in_branches():
    featex = fruits.Fruit()

    featex.add(fruits.preparation.INC)

    featex.add(fruits.core.SimpleWord("[11][1][221]"))
    featex.add(fruits.core.SimpleWord("[11][1][2][1]"))
    featex.add(fruits.core.SimpleWord("[12]"))
    featex.add(fruits.core.SimpleWord("[22]"))

    featex.add(fruits.sieving.PPV)
    featex.add(fruits.sieving.MAX)
    featex.add(fruits.sieving.MIN)

    featex.fork()

    featex.add(fruits.core.SimpleWord("[111][2]"))
    featex.add(fruits.core.SimpleWord("[1][22]"))
    featex.add(fruits.core.SimpleWord("[112][2]"))
    featex.add(fruits.core.SimpleWord("[1]"))
    featex.add(fruits.core.SimpleWord("[2]"))

    featex.add(fruits.sieving.PPV)
    featex.add(fruits.sieving.MAX)
    featex.add(fruits.sieving.MIN)

    assert featex.nfeatures() == 27

    featex.branch(0).calculator.mode = "extended"

    assert featex.nfeatures() == 42

    X = np.random.random_sample((100, 2, 1000))

    featex.fit(X)
    X_transform = featex.transform(X)

    assert X_transform.shape == (100, 42)

def test_branches():
    featex = fruits.Fruit()

    w1 = fruits.core.wording.SimpleWord("[1]")
    w2 = fruits.core.wording.SimpleWord("[2]")
    w3 = fruits.core.wording.SimpleWord("[11]")
    w4 = fruits.core.wording.SimpleWord("[12]")
    w5 = fruits.core.wording.SimpleWord("[1][1]")
    w6 = fruits.core.wording.SimpleWord("[1][2]")

    featex.add(w1, w2, w3)
    featex.add(fruits.sieving.MAX)
    featex.fork()
    featex.add(w4, w5, w6)
    featex.add(fruits.sieving.MIN)

    assert featex.nfeatures() == 6

    features = featex.fit_transform(X_1)

    assert features.shape == (2, 6)

    np.testing.assert_allclose(np.array([
                                    [1.8,3,50.64,-8,-24.6,-16.6],
                                    [21,-5,129,-44,0,-232.5]
                               ]),
                               features)
