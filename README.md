# FRUITS
(**F**eature Ext**R**action **U**sing **IT**erated **S**ums)<br>
The python package __FRUITS__ is a collection of transformations that allows the extraction of features from multidimensional time series data. These features can then be used in a classification.

## Installation
__FRUITS__ can be installed on your local machine by using the file [setup.py](setup.py).
```
  $ python setup.py install
```

## Documentation
The documentation of __FRUITS__ can be created by calling `make html` in the [docs](docs) folder. This will need a few dependencies to work. Please install the following packages using `pip` or `conda` before executing the `make` command.
- sphinx
- sphinx_rtd_theme

This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a browser to access the documentation.

## Pipeline
The main class in __FRUITS__ is `fruits.Fruit`. This object acts as a pipeline for the feature extraction and can be fully customized.<br>
Time series datasets go through different `fruits.FruitBranch` objects within the pipeline that transform the data with the following three steps.
- Data Preparation: `DataPreparateur` objects are used to preprocess the data. This is an optional step.
- Calculation of iterated sums: `Word` objects specify which _iterated sums_ should be calculated.<br>
  For example:<br>
  `<[11], ISS(X)>=numpy.cumsum([x^2 for x in X])` is the result of <br>
  `fruits.core.ISS(X, [fruits.words.SimpleWord("[11]")])`<br>
  The module ``fruits.signature`` together with ``fruits.words`` implements the *iterated sums signature* ISS.
  The definition and applications of that signature can be found in [this paper](https://link.springer.com/article/10.1007/s10440-020-00333-x>)
  by Diehl *et al.*.
- Feature Sieving: `FeatureSieve` objects extract single numerical values (i.e. features) from the arrays calculated in the previous step.<br>
  The total number of features per time series is the number of sieves times the number of words added to the `fruits.Fruit`.

A single _fruit_ can have multiple _fruit branches_. The features of each branch will be concatenated at the end of the extraction process.
  
## Example
A simple example could look like this:
```python
# think of a 3 dimensional time series dataset
X_train, y_train, X_test, y_test = ...

# create a Fruit object
myfruit = fruits.Fruit("myfruit - Fruit class example")

# add a DataPreparateur to it by using predefined ones from fruits.preparation
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
```

## UCR-Experiments
Have a look at the [instructions](experiments/README.md) for more information on how to execute some experiments with __FRUITS__.

## Unit tests
There are a bunch of [tests](tests) for __FRUITS__ available to execute. To do this, enter the command
```
  $ python -m pytest tests
```
in a terminal/command line from the main directory of this repository.
