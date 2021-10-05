from abc import ABC

import numpy as np

import fruits.requisites as reqs
from fruits.base.scope import force_input_shape


class FruitNode(ABC):
    """A FruitNode is the parent class for processing parts in a
    :class:`~fruits.base.fruit.Fruit` object, e.g. a
    :class:`~fruits.preparation.abstract.DataPreparateur` or a
    :class:`~fruits.sieving.abstract.FeatureSieve`.
    """

    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self._requisite = None
        self._req_container = None

    @property
    def name(self) -> str:
        """Simple representation string for this object without any
        computational meaning.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def set_requisite(self, requisite_ident: str):
        """Sets the requisite for this object. The identification has
        to be logged with :func:`~fruits.requisites.log`.

        :type requisite_ident: str
        """
        self._requisite = requisite_ident

    def _get_requisite(self, X: np.ndarray) -> np.ndarray:
        if self._requisite is None:
            return force_input_shape(X)
        elif self._req_container is None:
            return reqs.get(self._requisite).process(force_input_shape(X))
        return self._req_container.get(self._requisite)


    def _set_requisite_container(self, container: "RequisiteContainer"):
        self._req_container = container
