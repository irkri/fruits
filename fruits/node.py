from abc import ABC

import numpy as np

import fruits.requisites as requisites
from fruits.base.scope import force_input_shape

class FruitNode(ABC):
    """A FruitNode is the parent class for everything that can be added
    to a :class:~`fruits.base.fruit.Fruit`.
    """
    def __init__(self):
        super().__init__()
        self._requisite = None
        self._req_container = None

    def set_requisite(self, requisite_ident: str):
        """Sets the requisite for this object. The identification has
        to be logged with :func:~`fruits.requisites.log`.
        
        :type requisite_ident: str
        """
        self._requisite = requisite_ident

    def _get_requisite(self, X: np.ndarray) -> np.ndarray:
        if self._requisite is None:
            return force_input_shape(X)
        elif self._req_container is None:
            return requisites.get(self._requisite).process(force_input_shape(X))
        return self._req_container.get(self._requisite)

    def _set_requisite_container(self, container):
        self._req_container = container
