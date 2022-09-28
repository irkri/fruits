Code Documentation
==================

Getting Started
---------------
A new pipeline for feature extraction with **FRUITS** can be implemented by creating a new
:class:`~fruits.fruit.Fruit`. This class allows the transformation of a time series dataset. The
dataset has to be given as a `numpy <https://github.com/numpy/numpy>`_ array. The shape of
the array should be ``(n_series, n_dimensions, series_length)``.

If you don't want to build your own fruit but rather create one that suits your time series dataset
automatically, have a look at the method :meth:`fruits.builder.build`.

Categories
----------

.. toctree::
   :maxdepth: 4
   :glob:

   packages/fruits
   packages/build
   packages/preparation/*
   packages/words/*
   packages/signature/*
   packages/sieving/*
