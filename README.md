# FRUITS - **F**eature Ext**R**action **U**sing **IT**erated **S**ums

[Paper on Arxiv](https://arxiv.org/abs/2311.14549)

This repository contains the code for FRUITS, a Python package implementing a collection of
transformations that extract features from univariate or multivariate time series. Additionally, we
provide the following:

- A [documentation](docs) for FRUITS available to build using `sphinx`.
- A number of [unit tests](tests) for FRUITS.
- An extensive suite of algorithms to compare and analyze FRUITS pipelines, called
  [corbeille](experiments/corbeille).
- The explicit pipelines (or just *fruits*) used in our paper, see
  [fruit_general.py](experiments/fruit_general.py),
  [fruit_reduced.py](experiments/fruit_reduced.py) and [fruit_twi.py](experiments/fruit_twi.py).
- An [ipynb notebook](experiments/datasets.ipynb) containing code that analysis FRUITS on some
  datasets in the UCR archive, using corbeille and
  [classically](https://github.com/irkri/classically).

## Installation
Install __FRUITS__ by cloning the repository to your local machine and using `poetry` to install
the package and all dependencies in a new virtual environment. Alternatively, use instead `pip` to
install FRUITS inside an existing environment (*).

    $ git clone https://github.com/irkri/fruits
    $ cd fruits
    $ poetry install          (or: $ python -m pip install -e .)

Without cloning the repository, use instead:

    $ pip install git+https://github.com/alienkrieg/fruits

If an error occures, please try commenting out the line
```
    # corbeille = {path = "experiments/corbeille/", optional = true, develop = true}
```
in the file [pyproject.toml](/pyproject.toml).

## Documentation
The documentation of __FRUITS__ can be created by calling `make html` in the [docs](docs) folder.
This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a
browser to access the documentation.

The Python dependencies (`sphinx`, `sphinx-rtd-theme`) needed to execute the command are listed in
the [toml file](/pyproject.toml) as development dependencies.

## Pipeline
__FRUITS__ implements the class `fruits.Fruit`. A `Fruit` consists of at least one slice
(`fruits.FruitSlice`). A single slice consists of the following building blocks.

- **Preparateurs**: Preprocess the input time series.
- **ISS**: Calculate iterated sums for different semirings, weightings and words.<br>
  For example:<br>
  `<[11], ISS(X)>=numpy.cumsum([x^2 for x in X])` is the result of <br>
  `fruits.ISS([fruits.words.SimpleWord("[11]")]).fit_transform(X)`<br>
  The definition and applications of the *iterated sums signature* ISS can be found in [this paper](https://link.springer.com/article/10.1007/s10440-020-00333-x)
  by Diehl *et al.*.
- **Sieves**: Extract single numerical values (i.e. the final features) from the arrays calculated
  in the previous step.

All features of each _fruit slice_ will be concatenated at the end of the pipeline.

## Example
A simple example could look like this:
```python
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
```

## UCR Experiments
Have a look at the [instructions](experiments/README.md) for more information on how to execute
some experiments with __FRUITS__.

## Unit Tests
There are a bunch of [tests](tests) for __FRUITS__ available to execute. To do this, enter the command
```
  $ python -m pytest tests
```
in a terminal/command line from the main directory of this repository.
