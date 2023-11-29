Introduction
============

Installation
------------

You'll find an up-to-date installation guide at
`the repository of FRUITS <https://github.com/alienkrieg/fruits>`_.


What is FRUITS?
---------------

**FRUITS** (**F**\ eature Ext\ **R**\ action **U**\ sing **IT**\ erated **S**\ ums) is a Python
package implementing time series features extraction tools for machine learning. It is designed for
univariate and multivariate time series data.

Structure
---------

**FRUITS** implements the class :class:`~fruits.fruit.Fruit`. A fruit consists of at least one
:class:`slice <fruits.fruit.FruitSlice>`.

A single slice can have:

- **Preparateurs** ...
    are used to preprocess the data.

- **Words** ...
    are used to calculate iterated sums.
    For example::

        fruits.ISS([fruits.words.SimpleWord("[11]")]).fit_transform(X)

    calculates::

        numpy.cumsum([x^2 for x in X])

    The definition and applications of the *iterated sums signature* ISS can be found in
    `this paper <https://link.springer.com/article/10.1007/s10440-020-00333-x>`_ by Diehl *et al.*.

- **Sieves** ...
    extract single numerical values (i.e. the final features) from the arrays
    calculated in the previous step.

All features of each *fruit slice* will be concatenated at the end of the pipeline.

Simple Example
--------------

The following code block shows a simple example on how to use **FRUITS**.

.. code-block:: python

    import numpy
    import fruits
    # time series dataset: 200 time series of length 100 in 3 dimensions
    X_train = numpy.random.sample((200, 3, 100))

    # create a fruit
    fruit = fruits.Fruit("My Fruit")

    # add preparateurs (optional)
    fruit.add(fruits.preparation.INC)

    # configure the type of Iterated Sums Signature being used
    iss = fruits.ISS(
        fruits.words.of_weight(2, dim=3),
        mode=fruits.ISSMode.EXTENDED,
    )
    fruit.add(iss)

    # choose from a variety of sieves for feature extraction
    fruit.add(fruits.sieving.NPI(q=(0.5, 1.0)))
    fruit.add(fruits.sieving.END)

    # cut a new fruit slice without the INC preparateur
    fruit.cut()
    fruit.add(iss.copy())
    fruit.add(fruits.sieving.NPI)
    fruit.add(fruits.sieving.END)

    # fit the fruit to the data and extract all features
    fruit.fit(X_train)
    X_train_features = fruit.transform(X_train)
