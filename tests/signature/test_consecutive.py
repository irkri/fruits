import numpy as np

import fruits


def test_simple_consecutive():
    fruit = fruits.Fruit()

    iss1 = fruits.ISS(
        [fruits.words.SimpleWord("[12][1]"),
         fruits.words.SimpleWord("[1][32]"),
         fruits.words.SimpleWord("[11][121][3]")],
        mode=fruits.ISSMode.EXTENDED,
    )
    iss2 = fruits.ISS(
        [fruits.words.SimpleWord("[11]"),
         fruits.words.SimpleWord("[111]"),
         fruits.words.SimpleWord("[111][1][11]"),
         fruits.words.SimpleWord("[1][1][11]")],
        mode=fruits.ISSMode.EXTENDED,
    )
    fruit.add(iss1, iss2)
    fruit.add(fruits.sieving.MAX)
    fruit.add(fruits.sieving.END)

    assert fruit.nfeatures() == 98

    x = np.random.random_sample((50, 3, 100))
    xt = fruit.fit_transform(x)

    x1 = iss1.fit_transform(x)
    x2 = np.zeros((x1.shape[0]*7, 50, 100))
    for i, itsum in enumerate(x1):
        x2[i*7:(i+1)*7] = iss2.fit_transform(itsum[:, np.newaxis, :])

    np.testing.assert_allclose(xt[:, ::2], np.swapaxes(x2.max(axis=2), 0, 1))
    np.testing.assert_allclose(xt[:, 1::2], np.swapaxes(x2[:, :, -1], 0, 1))
