import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3],
     [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0],
     [-5, -1, -4, -0.5, -8]]
])


def test_mode_in_slices():
    fruit = fruits.Fruit()

    fruit.add(fruits.preparation.INC)

    fruit.add(fruits.words.SimpleWord("[11][1][221]"))
    fruit.add(fruits.words.SimpleWord("[11][1][2][1]"))
    fruit.add(fruits.words.SimpleWord("[12]"))
    fruit.add(fruits.words.SimpleWord("[22]"))

    fruit.add(fruits.sieving.PPV)
    fruit.add(fruits.sieving.MAX)
    fruit.add(fruits.sieving.MIN)

    fruit.cut()

    fruit.add(fruits.words.SimpleWord("[111][2]"))
    fruit.add(fruits.words.SimpleWord("[1][22]"))
    fruit.add(fruits.words.SimpleWord("[112][2]"))
    fruit.add(fruits.words.SimpleWord("[1]"))
    fruit.add(fruits.words.SimpleWord("[2]"))

    fruit.add(fruits.sieving.PPV)
    fruit.add(fruits.sieving.MAX)
    fruit.add(fruits.sieving.MIN)

    assert fruit.nfeatures() == 27

    fruit[0].configure(iss_mode="extended")

    assert fruit.nfeatures() == 36

    X = np.random.random_sample((100, 2, 1000))

    fruit.fit(X)
    X_transform = fruit.transform(X)

    assert X_transform.shape == (100, 36)


def test_slices():
    fruit = fruits.Fruit()

    w1 = fruits.words.SimpleWord("[1]")
    w2 = fruits.words.SimpleWord("[2]")
    w3 = fruits.words.SimpleWord("[11]")
    w4 = fruits.words.SimpleWord("[12]")
    w5 = fruits.words.SimpleWord("[1][1]")
    w6 = fruits.words.SimpleWord("[1][2]")

    fruit.add(w1, w2, w3)
    fruit.add(fruits.sieving.MAX)
    fruit.cut()
    fruit.add(w4, w5, w6)
    fruit.add(fruits.sieving.MIN)

    assert fruit.nfeatures() == 6

    features = fruit.fit_transform(X_1)

    assert features.shape == (2, 6)

    np.testing.assert_allclose(np.array([
        [1.8, 3, 50.64, -8, -24.6, -16.6],
        [21, -5, 129, -44, 0, -232.5]
    ]), features)
