from abc import ABC
from typing import Any

import numpy as np

from ..seed import Seed


class Preparateur(Seed, ABC):
    """Abstract class for a preparateur.

    A preparateur transforms three dimensional numpy arrays containing
    univariate or multivariate time series.
    """

    def _fit(self, X: np.ndarray) -> None:
        pass

    def __eq__(self, other: Any) -> bool:
        return False
