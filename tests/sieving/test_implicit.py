import pytest
import numpy as np

import fruits

X_1 = np.array([
    [[-4, 0.8, 0, 5, -3], [2, 1, 0, 0, -7]],
    [[5, 8, 2, 6, 0], [-5, -1, -4, -0.5, -8]]
])


def test_ppv():
    ppv_1 = fruits.sieving.PPV(quantile=0, constant=True)
    ppv_2 = fruits.sieving.PPV(quantile=0.5, constant=False,
                               sample_size=1)

    np.testing.assert_allclose(np.array([[3/5], [4/5]]),
                               ppv_1.fit_transform(X_1[0]))
    np.testing.assert_allclose(np.array([[1], [0]]),
                               ppv_2.fit_transform(X_1[1]))

    ppv_1_copy = ppv_1.copy()

    np.testing.assert_allclose(ppv_1.fit_transform(X_1[0]),
                               ppv_1_copy.fit_transform(X_1[0]))

    ppvc_1 = fruits.sieving.CPV(quantile=0, constant=True)

    np.testing.assert_allclose([[1/3], [0.0]], ppvc_1.fit_transform(X_1[0]))

    with pytest.raises(ValueError):
        fruits.sieving.PPV(quantile=[0.5, 0.1, -1],
                           constant=[False, True, False])

    with pytest.raises(ValueError):
        fruits.sieving.CPV(quantile=[0.5, 2], constant=False)

    ppv_group_1 = fruits.sieving.PPV(quantile=[0.5, 0.1, 0.7],
                                     constant=False,
                                     sample_size=1,
                                     segments=False)

    np.testing.assert_allclose(np.array([[1, 1, 3/5], [0, 4/5, 0]]),
                               ppv_group_1.fit_transform(X_1[1]))

    ppv_group_2 = fruits.sieving.PPV(quantile=[0.5, 0.1, 0.7],
                                     constant=False,
                                     sample_size=1,
                                     segments=True)

    np.testing.assert_allclose(np.array([[0, 2/5], [4/5, 0]]),
                               ppv_group_2.fit_transform(X_1[1]))

    ppv_group_3 = fruits.sieving.PPV(quantile=[-5, 0, 2],
                                     constant=True,
                                     sample_size=1,
                                     segments=False)

    np.testing.assert_allclose(np.array([[1, 1, 4/5], [4/5, 0, 0]]),
                               ppv_group_3.fit_transform(X_1[1]))

    ppv_group_4 = fruits.sieving.PPV(quantile=[0, -5, 2],
                                     constant=True,
                                     sample_size=1,
                                     segments=True)

    np.testing.assert_allclose(np.array([[0, 1/5], [4/5, 0]]),
                               ppv_group_4.fit_transform(X_1[1]))

    ppv_group_4_copy = ppv_group_4.copy()

    np.testing.assert_allclose(ppv_group_4.fit_transform(X_1[1]),
                               ppv_group_4_copy.fit_transform(X_1[1]))
