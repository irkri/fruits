from typing import Literal

import numpy as np

from . import preparation, sieving
from .fruit import Fruit
from .iss.iss import ISS, ISSMode
from .iss.words.creation import of_weight
from .iss.words.word import SimpleWord, Word
from .preparation.abstract import Preparateur
from .sieving.abstract import FeatureSieve


class UnivariateFruitBuilder:
    """Builds a :class:`~fruits.core.fruit.Fruit` suited for univariate
    time series.
    """

    def build(self, X: np.ndarray) -> Fruit:
        length = X.shape[2]
        fruit = Fruit("Built by UnivariateFruitBuilder")

        iss = self._choose_iss("leading")

        fruit.cut()
        fruit.add(preparation.INC)
        fruit.add(*self._choose_preparateurs("single", length))
        fruit.add(iss)
        fruit.add(*self._choose_sieves("small"))

        fruit.cut()
        fruit.add(*self._choose_preparateurs("single", length))
        fruit.add(iss)
        fruit.add(*self._choose_sieves("small"))

        smalliss = self._choose_iss("small")
        filters = self._choose_preparateurs("filter", length)
        for fltr in filters:
            fruit.cut()
            fruit.add(preparation.INC)
            fruit.add(fltr)
            fruit.add(smalliss)
            fruit.add(*self._choose_sieves("small"))

            fruit.cut()
            fruit.add(fltr)
            fruit.add(smalliss)
            fruit.add(*self._choose_sieves("small"))

        return fruit

    def _choose_preparateurs(
        self,
        mode: str,
        length: int,
    ) -> tuple[Preparateur, ...]:
        if mode == "single":
            return (
                preparation.DIM(
                    lambda X: np.expand_dims(
                        X[:, 0, :] * (X[:, 0, :] > 0), axis=1)
                ),
            )
        if mode == "filter":
            return tuple(
                preparation.DIL()
                for _ in range(int(10 * np.floor(np.log10(length))))
            )
        raise ValueError(f"Unknown mode supplied: {mode!r}")

    def _choose_iss(
        self,
        mode: str,
    ) -> ISS:
        # returns a list of words and a calculator mode
        if mode == "leading":
            words = of_weight(4, 1)
            preword = SimpleWord("[2]")
            leading_words = []
            for word in words:
                w = preword.copy()
                w.multiply(str(word))
                leading_words.append(w)
            return ISS(leading_words, mode=ISSMode.EXTENDED)
        if mode == "double":
            return ISS(of_weight(3, 2), mode=ISSMode.SINGLE)
        if mode == "small":
            return ISS(of_weight(3, 1), mode=ISSMode.EXTENDED)
        if mode == "large":
            return ISS(of_weight(4, 1), mode=ISSMode.EXTENDED)
        raise ValueError(f"Unknown mode supplied: {mode!r}")

    def _choose_sieves(self, size: str) -> list[FeatureSieve]:
        # returns the best working sieve configurations
        if size == "large":
            return [
                sieving.PPV([i/6 for i in range(1, 6)]),
                sieving.CPV([i/6 for i in range(1, 6)]),
                sieving.PIA([0.2, 0.4, 0.6, 0.8, -1]),
                sieving.MAX([1, 0.5, -1]),
                sieving.MIN([1, 0.5, -1]),
                sieving.END([i/10 for i in range(1, 10)]+[-1]),
            ]
        if size == "small":
            return [
                sieving.PPV([i/4 for i in range(1, 4)]),
                sieving.PIA([0.5, -1]),
                sieving.END([0.5, -1]),
            ]
        raise ValueError(f"Unknown size supplied: {size!r}")


class MultivariateFruitBuilder:
    """Builds a :class:`~fruits.core.fruit.Fruit` suited for
    multivariate time series.
    """

    def build(self, X: np.ndarray) -> Fruit:
        fruit = Fruit("Built by MultivariateFruitBuilder")

        dim = X.shape[1]
        iss = self._choose_iss(dim)
        sieves = self._choose_sieves(dim)

        fruit.add(preparation.INC)
        fruit.add(iss)
        fruit.add(*sieves)

        fruit.cut()
        fruit.add(iss)
        fruit.add(*sieves)

        return fruit

    def _choose_iss(
        self,
        dim: int,
    ) -> ISS:
        # chooses fitting words, calculator mode based on dimensionality
        if 2 <= dim <= 3:
            return ISS(of_weight(6 - dim, dim), mode=ISSMode.EXTENDED)
        if 4 <= dim <= 18:
            return ISS(of_weight(2, dim), mode=ISSMode.EXTENDED)
        if 19 <= dim <= 47:
            words = []
            for d in range(1, 4):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
                        words.append(SimpleWord(f"[({i})({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return ISS(words, mode=ISSMode.EXTENDED)
        if 48 <= dim <= 100:
            words = []
            for d in range(1, 5):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return ISS(words, mode=ISSMode.EXTENDED)
        words = []
        for i in range(dim):
            if i + 1 <= dim:
                words.append(SimpleWord(f"[({i})][({i+1})]"))
        words.append(SimpleWord(f"[({dim})]"))
        return ISS(words, mode=ISSMode.EXTENDED)

    def _choose_sieves(self, dim: int) -> tuple[FeatureSieve, ...]:
        # chooses fitting sieves based on dimensionality
        if 2 <= dim <= 3:
            return (
                sieving.PPV([0.25, 0.5, 0.75]),
                sieving.PIA(),
                sieving.MAX(),
                sieving.MIN(),
                sieving.END([0.5, -1]),
            )
        if 4 <= dim <= 8:
            return (
                sieving.PPV(),
                sieving.PIA([0.2, 0.4, 0.6, 0.8, -1]),
                sieving.MAX(),
                sieving.MIN(),
                sieving.END(),
            )
        if 9 <= dim <= 18:
            return (
                sieving.PPV(),
                sieving.END(),
            )
        if 19 <= dim <= 47:
            return (
                sieving.PPV(),
                sieving.PIA(),
                sieving.END(),
            )
        return (
            sieving.PPV(),
            sieving.END(),
        )


def build(X: np.ndarray) -> Fruit:
    """Builds a :class:`~fruits.core.fruit.Fruit` based on the given
    time series dataset.

    The returned fruit is expected to be a good candidate for the
    classification of the given dataset.

    Args:
        X (np.ndarray): Three dimensional array with the shape
            ``(n_time_series, n_dimensions, series_length)``.
    """
    if X.shape[1] == 1:
        return UnivariateFruitBuilder().build(X)
    return MultivariateFruitBuilder().build(X)
