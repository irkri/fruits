from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np

from fruits.base.fruit import Fruit
from fruits.base.scope import force_input_shape
from fruits.core.wording import Word, SimpleWord
from fruits.core.generation import simplewords_by_weight

from fruits.preparation.abstract import DataPreparateur
from fruits.sieving.abstract import FeatureSieve
from fruits import preparation
from fruits import sieving


class FruitBuilder(ABC):
    """Abstract class that is inherited by classes which are building
    :class:`~fruits.base.fruit.Fruit` objects.
    """

    @abstractmethod
    def build(self, X: np.ndarray) -> Fruit:
        """Builds a Fruit based on the given dataset.

        :param X_train: Three dimensional numpy array containing
            multidimensional time series data.
        :type X_train: np.ndarray
        :rtype: Fruit
        """


class UnivariateFruitBuilder(FruitBuilder):
    """Class that builds a :class:`~fruits.base.fruit.Fruit` object for
    a given multivariate time series dataset.

    The returned fruit is expected to be a good candidate for the
    classification of the given dataset.
    """

    def build(self, X_train: np.ndarray) -> Fruit:
        """Builds a Fruit based on the given dataset.

        :param X_train: Three dimensional numpy array containing
            multidimensional time series data.
        :type X_train: np.ndarray
        :rtype: Fruit
        """
        X_train = force_input_shape(X_train)
        length = X_train.shape[2]
        fruit = Fruit("Built by UnivariateFruitBuilder")

        leadingwords, mode = self._choose_words("leading")

        fruit.fork()
        fruit.add(preparation.transform.INC)
        fruit.add(self._choose_preparateurs("single", length))
        fruit.add(leadingwords)
        fruit.branch().calculator.mode = mode
        fruit.add(self._choose_sieves("small"))

        fruit.fork()
        fruit.add(self._choose_preparateurs("single", length))
        fruit.add(leadingwords)
        fruit.branch().calculator.mode = mode
        fruit.add(self._choose_sieves("small"))

        smallwords, mode = self._choose_words("small")
        filters = self._choose_preparateurs("filter", length)
        for fltr in filters:
            fruit.fork()
            fruit.add(preparation.transform.INC)
            fruit.add(fltr)
            fruit.add(smallwords)
            fruit.branch().calculator.mode = mode
            fruit.add(self._choose_sieves("small"))

            fruit.fork()
            fruit.add(fltr)
            fruit.add(smallwords)
            fruit.branch().calculator.mode = mode
            fruit.add(self._choose_sieves("small"))

        return fruit

    def _choose_preparateurs(self,
                             mode: str,
                             length: int) -> List[DataPreparateur]:
        if mode == "single":
            return [
                preparation.dimension.DIM(
                    lambda X: np.expand_dims(
                        X[:, 0, :] * (X[:, 0, :] > 0), axis=1)
                ),
            ]
        elif mode == "filter":
            filters = []
            n = int(10 * np.floor(np.log10(length)))
            for i in range(n):
                filters.append(preparation.filter.DIL())
            return filters

    def _choose_words(self, mode: str) -> Tuple[List[Word], str]:
        # returns a list of words and a calculator mode
        if mode == "leading":
            words = simplewords_by_weight(4, 1)
            preword = SimpleWord("[2]")
            leading_words = []
            for word in words:
                w = preword.copy()
                w.multiply(str(word))
                leading_words.append(w)
            return leading_words, "extended"
        elif mode == "double":
            return simplewords_by_weight(3, 2), "single"
        elif mode == "small":
            return simplewords_by_weight(3, 1), "extended"
        elif mode == "large":
            return simplewords_by_weight(4, 1), "extended"

    def _choose_sieves(self, size: str) -> List[FeatureSieve]:
        # returns the best working sieve configurations
        if size == "large":
            return [
                sieving.implicit.PPV([i/6 for i in range(1, 6)]),
                sieving.implicit.CPV([i/6 for i in range(1, 6)]),
                sieving.explicit.PIA([0.2, 0.4, 0.6, 0.8, -1]),
                sieving.explicit.MAX([1, 0.5, -1], segments=True),
                sieving.explicit.MIN([1, 0.5, -1], segments=True),
                sieving.explicit.END([i/10 for i in range(1, 10)]+[-1]),
            ]
        elif size == "small":
            return [
                sieving.implicit.PPV([i/4 for i in range(1, 4)]),
                sieving.explicit.PIA([0.5, -1]),
                sieving.explicit.END([0.5, -1]),
            ]


class MultivariateFruitBuilder(FruitBuilder):
    """Class that builds a :class:`~fruits.base.fruit.Fruit` object out
    of a given multivariate time series dataset.

    The returned fruit is expected to be a good candidate for the
    classification of the given dataset.
    """

    def build(self, X_train: np.ndarray) -> Fruit:
        """Builds a Fruit based on the given dataset.

        :param X_train: Three dimensional numpy array containing
            multidimensional time series data.
        :type X_train: np.ndarray
        :rtype: Fruit
        """
        X_train = force_input_shape(X_train)
        fruit = Fruit("Built by MultivariateFruitBuilder")

        dim = X_train.shape[1]
        words, mode = self._choose_words(dim)
        sieves = self._choose_sieves(dim)

        fruit.add(preparation.transform.INC)
        fruit.add(words)
        fruit.branch().calculator.mode = mode
        fruit.add(sieves)

        fruit.fork()
        fruit.add(words)
        fruit.branch().calculator.mode = mode
        fruit.add(sieves)

        return fruit

    def _choose_words(self, dim: int) -> Tuple[List[Word], str]:
        # chooses fitting words, calculator mode based on dimensionality
        if 2 <= dim <= 3:
            return simplewords_by_weight(6 - dim, dim), "extended"
        elif 4 <= dim <= 18:
            return simplewords_by_weight(2, dim), "extended"
        elif 19 <= dim <= 47:
            words = []
            for d in range(1, 4):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
                        words.append(SimpleWord(f"[({i})({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return words, "extended"
        elif 48 <= dim <= 100:
            words = []
            for d in range(1, 5):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return words, "extended"
        elif 100 < dim:
            words = []
            for i in range(dim):
                if i + 1 <= dim:
                    words.append(SimpleWord(f"[({i})][({i+1})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return words, "extended"

    def _choose_sieves(self, dim: int) -> List[FeatureSieve]:
        # chooses fitting sieves based on dimensionality
        if 2 <= dim <= 3:
            return [
                sieving.implicit.PPV([0.25, 0.5, 0.75]),
                sieving.explicit.PIA(),
                sieving.explicit.MAX(),
                sieving.explicit.MIN(),
                sieving.explicit.END([0.5, -1]),
            ]
        elif 4 <= dim <= 8:
            return [
                sieving.implicit.PPV(),
                sieving.explicit.PIA([0.2, 0.4, 0.6, 0.8, -1]),
                sieving.explicit.MAX(),
                sieving.explicit.MIN(),
                sieving.explicit.END(),
            ]
        elif 9 <= dim <= 18 or 48 <= dim:
            return [
                sieving.implicit.PPV(),
                sieving.explicit.END(),
            ]
        elif 19 <= dim <= 47:
            return [
                sieving.implicit.PPV(),
                sieving.explicit.PIA(),
                sieving.explicit.END(),
            ]


def build(X_train: np.ndarray) -> Fruit:
    """Builds a :class:`~fruits.base.fruit.Fruit` object based on a
    given time series dataset.

    The returned fruit is expected to be a good candidate for the
    classification of the given dataset.

    :param X: Time series dataset (preferably the training set).
        This should be a three dimensional numpy array. Check
        :meth:`~fruits.base.scope.force_input_shape`.
    :type X: np.ndarray
    :rtype: Fruit
    """
    X_train = force_input_shape(X_train)
    if X_train.shape[1] == 1:
        return UnivariateFruitBuilder().build(X_train)
    else:
        return MultivariateFruitBuilder().build(X_train)
