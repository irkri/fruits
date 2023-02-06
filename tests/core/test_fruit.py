import numpy as np

import fruits

X_1 = np.array([
    [[-4., .8, 0., 5., -3.], [2., 1., 0., 0., -7.]],
    [[5., 8., 2., 6., 0.], [-5., -1., -4., -.5, -8.]]
])


def test_n_features():
    fruit = fruits.Fruit()

    fruit.add(fruits.preparation.INC(zero_padding=False))

    fruit.add(fruits.ISS(fruits.words.of_weight(4, dim=2)))

    assert len(fruit.get_slice().get_iss()[0].words) == 82

    fruit.add(fruits.sieving.PPV(quantile=0, constant=True))
    fruit.add(fruits.sieving.PPV(quantile=0.2, constant=False,
                                 sample_size=1))
    fruit.add(fruits.sieving.PPV(quantile=[0.2, 5], constant=[False, True],
                                 sample_size=1, segments=True))
    fruit.add(fruits.sieving.MAX(cut=[0.1, 0.5, 0.9]))
    fruit.add(fruits.sieving.MIN(cut=[0.1, 0.5, 0.9]))

    assert fruit.nfeatures() == 574

    fruit_copy = fruit.deepcopy()

    assert fruit_copy.nfeatures() == 574

    fruit.fit_transform(X_1)

    sieves = fruit.get_slice()._sieves_extended
    for word_sieves in sieves:
        for i in range(len(word_sieves) - 1):
            assert word_sieves[i]._cache is word_sieves[i+1]._cache

    del fruit
