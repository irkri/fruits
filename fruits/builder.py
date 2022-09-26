from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from . import preparation, sieving
from .fruit import Fruit
from .iss.words.creation import of_weight
from .iss.words.word import SimpleWord, Word
from .preparation.abstract import Preparateur
from .sieving.abstract import FeatureSieve


class FruitBuilder(ABC):

    @abstractmethod
    def build(self, X: np.ndarray) -> Fruit:
        """Builds and returns a :class:`~fruits.core.fruit.Fruit` based
        on the given dataset.

        Args:
            X (np.ndarray): Three dimensional array with the shape
                ``(n_time_series, n_dimensions, series_length)``.
        """


class UnivariateFruitBuilder(FruitBuilder):
    """Builds a :class:`~fruits.core.fruit.Fruit` suited for univariate
    time series.
    """

    def build(self, X: np.ndarray) -> Fruit:
        length = X.shape[2]
        fruit = Fruit("Built by UnivariateFruitBuilder")

        leadingwords, mode = self._choose_words("leading")

        fruit.fork()
        fruit.add(preparation.INC)
        fruit.add(*self._choose_preparateurs("single", length))
        fruit.add(*leadingwords)
        fruit.branch().configure(iss_mode=mode)
        fruit.add(*self._choose_sieves("small"))

        fruit.fork()
        fruit.add(*self._choose_preparateurs("single", length))
        fruit.add(*leadingwords)
        fruit.branch().configure(iss_mode=mode)
        fruit.add(*self._choose_sieves("small"))

        smallwords, mode = self._choose_words("small")
        filters = self._choose_preparateurs("filter", length)
        for fltr in filters:
            fruit.fork()
            fruit.add(preparation.INC)
            fruit.add(fltr)
            fruit.add(*smallwords)
            fruit.branch().configure(iss_mode=mode)
            fruit.add(*self._choose_sieves("small"))

            fruit.fork()
            fruit.add(fltr)
            fruit.add(*smallwords)
            fruit.branch().configure(iss_mode=mode)
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
        elif mode == "filter":
            return tuple(
                preparation.DIL()
                for _ in range(int(10 * np.floor(np.log10(length))))
            )
        raise ValueError(f"Unknown mode supplied: {mode!r}")

    def _choose_words(
        self,
        mode: str,
    ) -> tuple[tuple[Word, ...], Literal["single", "extended"]]:
        # returns a list of words and a calculator mode
        if mode == "leading":
            words = of_weight(4, 1)
            preword = SimpleWord("[2]")
            leading_words = []
            for word in words:
                w = preword.copy()
                w.multiply(str(word))
                leading_words.append(w)
            return tuple(leading_words), "extended"
        elif mode == "double":
            return of_weight(3, 2), "single"
        elif mode == "small":
            return of_weight(3, 1), "extended"
        elif mode == "large":
            return of_weight(4, 1), "extended"
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
        elif size == "small":
            return [
                sieving.PPV([i/4 for i in range(1, 4)]),
                sieving.PIA([0.5, -1]),
                sieving.END([0.5, -1]),
            ]
        raise ValueError(f"Unknown size supplied: {size!r}")


class MultivariateFruitBuilder(FruitBuilder):
    """Builds a :class:`~fruits.core.fruit.Fruit` suited for
    multivariate time series.
    """

    def build(self, X: np.ndarray) -> Fruit:
        fruit = Fruit("Built by MultivariateFruitBuilder")

        dim = X.shape[1]
        words, mode = self._choose_words(dim)
        sieves = self._choose_sieves(dim)

        fruit.add(preparation.INC)
        fruit.add(*words)
        fruit.branch().configure(iss_mode=mode)
        fruit.add(*sieves)

        fruit.fork()
        fruit.add(*words)
        fruit.branch().configure(iss_mode=mode)
        fruit.add(*sieves)

        return fruit

    def _choose_words(
        self,
        dim: int,
    ) -> tuple[tuple[Word, ...], Literal["single", "extended"]]:
        # chooses fitting words, calculator mode based on dimensionality
        if 2 <= dim <= 3:
            return of_weight(6 - dim, dim), "extended"
        elif 4 <= dim <= 18:
            return of_weight(2, dim), "extended"
        elif 19 <= dim <= 47:
            words = []
            for d in range(1, 4):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
                        words.append(SimpleWord(f"[({i})({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return tuple(words), "extended"
        elif 48 <= dim <= 100:
            words = []
            for d in range(1, 5):
                for i in range(1, dim + 1):
                    if i + d <= dim:
                        words.append(SimpleWord(f"[({i})][({i+d})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return tuple(words), "extended"
        else:
            words = []
            for i in range(dim):
                if i + 1 <= dim:
                    words.append(SimpleWord(f"[({i})][({i+1})]"))
            words.append(SimpleWord(f"[({dim})]"))
            return tuple(words), "extended"

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
        elif 4 <= dim <= 8:
            return (
                sieving.PPV(),
                sieving.PIA([0.2, 0.4, 0.6, 0.8, -1]),
                sieving.MAX(),
                sieving.MIN(),
                sieving.END(),
            )
        elif 9 <= dim <= 18:
            return (
                sieving.PPV(),
                sieving.END(),
            )
        elif 19 <= dim <= 47:
            return (
                sieving.PPV(),
                sieving.PIA(),
                sieving.END(),
            )
        else:
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
    else:
        return MultivariateFruitBuilder().build(X)
