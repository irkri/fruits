Code Documentation
==================

Getting Started
---------------
To get started, have a look at the definition of the main class in **FRUITS**,
*the* :class:`~fruits.core.fruit.Fruit`.

This class allows the transformation of a time series dataset. The dataset has
to be given as a `numpy <https://github.com/numpy/numpy>`_ array. The shape of
the array is explained in :meth:`fruits.scope.force_input_shape`.

Categories
----------

.. toctree::
   :maxdepth: 4
   :glob:

   packages/core/fruit
   packages/core/builder
   packages/preparation/*
   packages/words/*
   packages/signature/*
   packages/sieving/*
   packages/core/callback
   packages/*
