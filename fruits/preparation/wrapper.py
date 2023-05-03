__all__ = ["DIM", "NEW"]

from collections.abc import Sequence
from typing import Union

import numpy as np

from .abstract import Preparateur


class DIM(Preparateur):
    """Wrapper for another preparateur. DIM will evaluate the given
    preparateur only on the specified dimension(s) of the input time
    series and only replaces these dimensions by the transformed ones.

    Args:
        preparateur (Preparateur): A preparateur that is used for the
            transform.
        dim (int or sequence of int): One or more integers specifying
            which dimensions of the input time series to transform.
    """

    def __init__(
        self,
        preparateur: Preparateur,
        dim: Union[int, Sequence[int]],
    ) -> None:
        self._preparateur = preparateur
        self._dim = np.array([dim]) if isinstance(dim, int) else np.array(dim)

    @property
    def requires_fitting(self) -> bool:
        return self._preparateur.requires_fitting

    def _fit(self, X: np.ndarray) -> None:
        self._preparateur.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        result = X.copy()
        transformed = self._preparateur.transform(X[:, self._dim, :])
        result[:, self._dim, :] = transformed
        return result

    def _copy(self) -> "DIM":
        return DIM(self._preparateur.copy(), tuple(self._dim))

    def __str__(self) -> str:
        return f"DIM({str(self._preparateur)}, {tuple(self._dim)})"


class NEW(Preparateur):
    """Wrapper for another preparateur. NEW will evaluate the given
    preparateur and adds the results as new dimensions to the input time
    series.

    Args:
        preparateur (Preparateur): A preparateur which results will be
            appended to the input dimensions.
    """

    def __init__(self, preparateur: Preparateur) -> None:
        self._preparateur = preparateur

    @property
    def requires_fitting(self) -> bool:
        return self._preparateur.requires_fitting

    def _fit(self, X: np.ndarray) -> None:
        self._preparateur.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        transformed = self._preparateur.transform(X)
        result = np.zeros(
            (X.shape[0], X.shape[1]+transformed.shape[1], X.shape[2]),
            dtype=np.float32,
        )
        result[:, :X.shape[1], :] = X.copy()
        result[:, X.shape[1]:, :] = transformed
        return result

    def _copy(self) -> "NEW":
        return NEW(self._preparateur.copy())

    def __str__(self) -> str:
        return f"NEW({str(self._preparateur)})"
