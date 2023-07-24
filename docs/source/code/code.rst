Code Documentation
==================

Getting Started
---------------
A new pipeline for feature extraction with **FRUITS** can be implemented by creating a new
:class:`~fruits.fruit.Fruit`. This class allows the transformation of a time series dataset. The
dataset has to be given as a `numpy <https://github.com/numpy/numpy>`_ array. The shape of
the array should be ``(n_series, n_dimensions, series_length)``.

Categories
----------

.. toctree::
   :maxdepth: 4
   :glob:

   packages/fruits
   packages/preparation/*
   packages/words/*
   packages/signature/*
   packages/sieving/*
