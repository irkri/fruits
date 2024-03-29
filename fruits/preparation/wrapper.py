__all__ = ["DIM", "NEW"]

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from .abstract import Preparateur


class DIM(Preparateur):
    """Wrapper for another preparateur. DIM will evaluate the given
    preparateur only on the specified dimension(s) of the input time
    series. The results are appended to the original time series with
    the given original dimensions removed. This way, the time series
    might get reordered!

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
        self._preparateur.fit(X[:, self._dim, :])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        transformed = self._preparateur.transform(X[:, self._dim, :])
        result = np.delete(X, self._dim, axis=1)
        result = np.concatenate((result, transformed), axis=1)
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
        preparateur (Preparateur, optional): A preparateur of which
            results will be appended to the input dimensions. Defaults
            to None, in which case all current dimensions will be copied
            and appended to the input time series.
    """

    def __init__(self, preparateur: Optional[Preparateur] = None) -> None:
        self._preparateur = preparateur

    @property
    def requires_fitting(self) -> bool:
        if self._preparateur is None:
            return False
        return self._preparateur.requires_fitting

    def _fit(self, X: np.ndarray) -> None:
        if self._preparateur is not None:
            self._preparateur.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._preparateur is None:
            result = np.zeros(
                (X.shape[0], 2*X.shape[1], X.shape[2]),
                dtype=X.dtype,
            )
            result[:, :X.shape[1], :] = X[:, :, :]
            result[:, X.shape[1]:, :] = X[:, :, :]
            return result

        transformed = self._preparateur.transform(X)
        result = np.zeros(
            (X.shape[0], X.shape[1]+transformed.shape[1], X.shape[2]),
            dtype=X.dtype,
        )
        result[:, :X.shape[1], :] = X[:, :, :]
        result[:, X.shape[1]:, :] = transformed
        return result

    def _copy(self) -> "NEW":
        if self._preparateur is None:
            return NEW()
        return NEW(self._preparateur.copy())

    def __str__(self) -> str:
        return f"NEW({str(self._preparateur)})"
