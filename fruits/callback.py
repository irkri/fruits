from abc import ABC

import numpy as np


class AbstractCallback(ABC):
    """Another class inheriting ``AbstractCallback`` can overwrite one
    or more of the class methods.

    The callback can then be used in a call of
    :meth:`fruits.core.fruit.Fruit.transform`.
    """

    def on_next_branch(self) -> None:
        """Called every time the current
        :class:`~fruits.core.fruit.FruitBranch` in a
        :class:`~fruits.core.fruit.Fruit` object is switched.
        """

    def on_preparateur(self, X: np.ndarray) -> None:
        """Called after the calculation of prepared data for each
        :class:`~fruits.preparation.abstract.DataPreparateur`.
        """

    def on_preparation_end(self, X: np.ndarray) -> None:
        """Called once after the calculation of the prepared data with
        the last :class:`~fruits.preparation.abstract.DataPreparateur`.
        """

    def on_iterated_sum(self, X: np.ndarray) -> None:
        """Called for every iterated sum calculated for each single
        :class:`~fruits.words.word.Word`.
        """

    def on_sieve(self, X: np.ndarray) -> None:
        """Called after each use of a
        :class:`~fruits.sieving.abstract.FeatureSieve`.
        """

    def on_sieving_end(self, X: np.ndarray) -> None:
        """Called once at the end of the feature calculation."""
