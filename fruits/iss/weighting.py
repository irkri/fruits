__all__ = ["Weighting", "L1", "L2", "Indices", "Plateaus"]

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from ..cache import CacheType, SharedSeedCache


class Weighting(ABC):
    """Abstract class for exponential penalization for the calculation
    of iterated sums. Sums that use multiplications of time steps that
    are further apart from each other are scaled down exponentially. For
    two time steps ``i`` and ``j`` in the iterated sum, the summand is
    scaled by::

        e^(a*(g(j)-g(i)))

    where ``a`` is a given scalar. This scalar can be specified in a
    list of floats, each single float being applied to two consecutive
    indices for consecutive extended letters in words used by the
    iterated sum. An appropriate number of scalars have to be specified,
    matching or exceeding the length of the longest word in the
    :class:`ISS`. The function ``g`` transforming time steps is chosen
    by using on of the subclasses of this class.

    Args:
        scalars (sequence of float, optional): The float values used to
            scale the time index differences. If None is given, all
            scalars are assumed to be 1.
    """

    _cache: SharedSeedCache

    def __init__(
        self,
        scalars: Optional[Sequence[float]] = None,
    ) -> None:
        if scalars is not None:
            self._scalars = np.array(scalars, dtype=np.float32)
        else:
            self._scalars = None

    def get_fast_args(
        self,
        n: int,
        l: int,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        return self._scalars, self._get_lookup(n, l)

    @abstractmethod
    def _get_lookup(self, n: int, l: int) -> np.ndarray:
        ...


class Indices(Weighting):
    """Weighting for iterated sums where ``g(i)=i/N`` and ``N`` is the
    time series length. See class :class:`Weighting` for more
    information.

    Args:
        relative (bool, optional): Whether to scale the time steps into
            the interval ``[0,1]``. If set to False, instead use the
            identity ``g(i)=i``. Defaults to True.
    """

    def __init__(
        self,
        relative: bool = True,
        scalars: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(scalars=scalars)
        self._relative = relative

    def _get_lookup(self, n: int, l: int) -> np.ndarray:
        range_ = np.arange(l)
        if self._relative:
            range_ = range_ / l
        return np.ones((n, l)) * range_


class L1(Weighting):
    """Weighting for iterated sums where ``g(i)`` is the sum of absolute
    increments of the input time series up to time step ``i``. See class
    :class:`Weighting` for more information.
    """

    def _get_lookup(self, n: int, l: int) -> np.ndarray:
        return self._cache.get(CacheType.ISS, "L1")


class L2(Weighting):
    """Weighting for iterated sums where ``g(i)`` is the sum of squared
    increments of the input time series up to time step ``i``. See class
    :class:`Weighting` for more information.
    """

    def _get_lookup(self, n: int, l: int) -> np.ndarray:
        return self._cache.get(CacheType.ISS, "L2")


class Plateaus(Weighting):
    """Weighting for iterated sums where ``g`` is a step function. The
    steps are equally distributed. See class :class:`Weighting` for more
    information.

    Args:
        n (int): Number of plateaus. This results in a function
            comprised of ``n-1`` steps. Has to be ``> 1``.
    """

    def __init__(
        self,
        n: int,
        scalars: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(scalars=scalars)
        if n <= 1:
            raise ValueError(f"Number of plateaus ({n}) has to be > 1")
        self._nplateaus = n

    def _get_lookup(self, n: int, l: int) -> np.ndarray:
        range_ = np.ones(l)
        step = int(l/(self._nplateaus))
        for i in range(self._nplateaus):
            range_[i*step:(i+1)*step] = i / (self._nplateaus-1)
        return np.ones((n, l)) * range_
