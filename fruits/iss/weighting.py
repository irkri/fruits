from typing import Literal, Optional, Sequence

import numpy as np

from ..cache import CacheType, SharedSeedCache


class Weighting:
    """Exponential penalization for the calculation of iterated sums.
    Sums that use multiplications of time steps that are further apart
    from each other are scaled down exponentially. For two time steps
    ``i`` and ``j`` in the iterated sum, the summand is scaled by::

        e^(a*(j-i))

    where ``a`` is a given scalar. This scalar can be specified in a
    list of floats, each single float being applied to two consecutive
    indices for consecutive extended letters in words used by the
    iterated sum. An appropriate number of scalars have to be specified,
    matching or exceeding the length of the longest word in the
    :class:`ISS`.

    Args:
        scalars (sequence of float, optional): The float values used to
            scale the time index differences. If None is given, all
            scalars are assumed to be 1.
        use_sum ("L1" or "L2", optional): If this argument is supplied,
            instead of the original indices, the sum of the norms of
            time series values is used in the weighting. For example,
            "L1" refers to the sum of absolute values of increments.
            "L2" is the sum of squared increments up to the given time
            step.
    """

    _cache: SharedSeedCache

    def __init__(
        self,
        scalars: Optional[Sequence[float]] = None,
        use_sum: Optional[Literal["L1", "L2"]] = None,
    ) -> None:
        if scalars is not None:
            self._scalars = np.array(scalars, dtype=np.float32)
        else:
            self._scalars = None
        self._norm = use_sum

    def get_fast_args(
        self,
        n: int,
        l: int,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        if self._norm is not None:
            lookup = self._cache.get(CacheType.ISS, self._norm)
        else:
            lookup = np.ones((n, l)) * np.arange(l)
        return self._scalars, lookup
