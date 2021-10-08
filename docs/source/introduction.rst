Introduction
============

Installation
------------

For a quick and up-to-date installation, please go to
`the github page for FRUITS <https://github.com/alienkrieg/fruits>`_
and clone the repository. Then install using the ``setuptools`` package by
executing the following command in the cloned folder.

	>>> python setup.py install

What is FRUITS?
---------------

**FRUITS** (**F**\ eature Ext\ **R**\ action **U**\ sing **IT**\ erated **S**\ ums) is a
python package and a tool for machine learning. It is designed for feature extraction from multidimensional
time series data. These calculated features can then used for a classification task.

Structure
---------

The main class in **FRUITS** is ``fruits.Fruit``. This object acts as a pipeline for the feature extraction and can be fully customized.
Time series datasets go through different ``fruits.FruitBranch`` objects within the pipeline that transform the data with the following three steps.

A single *fruit* can have multiple *fruit branches*. The features of each branch will be concatenated at the end of the extraction process.

Data preparation / preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``DataPreparateur`` objects are used to preprocess the data.
This is an optional step.

Calculation of Iterated Sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Words specify which *iterated sums* should be calculated.
For example::

	<[11], ISS(X)>=numpy.cumsum([x^2 for x in X])

is the result of::

	fruits.signature.ISS(X, [fruits.words.SimpleWord("[11]")])

The module ``fruits.signature`` together with ``fruits.words`` implements the *iterated sums signature* ISS.
The definition and applications of that signature can be found in `this paper <https://link.springer.com/article/10.1007/s10440-020-00333-x>`_
by Diehl *et al.*.

Feature Sieving:
^^^^^^^^^^^^^^^^
``FeatureSieve`` objects extract single numerical values (i.e. features) from the arrays calculated in the previous step.
The total number of features per time series is ::

	[number of sieves] * [number of words]

Simple Example
--------------

The following code block shows a simple example on how to use **FRUITS**.

.. code-block:: python

	# think of a 3 dimensional time series dataset
	X_train, y_train, X_test, y_test = ...

	# create a Fruit object
	myfruit = fruits.Fruit("myfruit - Fruit class example")

	# add a DataPreparateur to it by using predefined ones from fruits.preparateurs
	myfruit.add(fruits.preparation.INC)

	# generate SimpleWord objects
	simplewords = fruits.words.simplewords_by_weight(2, dim=3)
	# simplewords is now the list of all words of weight 2 in 3 dimensions

	# add the words to the class instance
	myfruit.add(simplewords)

	# choose from a variety of FeatureSieve objects in fruits.sieving
	myfruit.add(fruits.sieving.PPV(quantile=0.5, constant=False))
	myfruit.add(fruits.sieving.MAX)

	# fork a new branch without preparateurs
	myfruit.fork()
	myfruit.add(simplewords)
	myfruit.add(fruits.sieving.PPV(quantile=0, constant=True))
	myfruit.add(fruits.sieving.MIN)

	# fit the object to the training data
	myfruit.fit(X_train)
	# get features for the training set
	X_train_features = myfruit.transform(X_train)
	# get features for the testing set
	X_test_features = myfruit.transform(X_test)
