# FRUITS
(**F**eature Ext**R**action **U**sing **IT**erated **S**ums)<br>
... is a collection of transformations that extract features from univariate or multivariate time series.

## Installation
Install __FRUITS__ by cloning the repository to your local machine and executing

    >>> python -m pip install .

in the main directory of this repository. If an error occures, please try commenting out the line
```
    # corbeille = {path = "experiments/corbeille/", optional = true, develop = true}
```
in the file [pyproject.toml](/pyproject.toml).

## Documentation
The documentation of __FRUITS__ can be created by calling `make html` in the [docs](docs) folder. This will need a few dependencies to work. Please install the following packages using `pip` or `conda` before executing the `make` command.
- sphinx
- sphinx_rtd_theme

This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a browser to access the documentation.

## Pipeline
__FRUITS__ implements the class `fruits.Fruit`. A `Fruit` consists of at least one slice (`fruits.FruitSlice`). A single slice can have
- **Preparateurs** ... are used to preprocess the data.
- **Words** ... are used to calculate iterated sums.<br>
  For example:<br>
  `<[11], ISS(X)>=numpy.cumsum([x^2 for x in X])` is the result of <br>
  `fruits.ISS([fruits.words.SimpleWord("[11]")]).fit_transform(X)`<br>
  The definition and applications of the *iterated sums signature* ISS can be found in [this paper](https://link.springer.com/article/10.1007/s10440-020-00333-x)
  by Diehl *et al.*.
- **Sieves** ... extract single numerical values (i.e. the final features) from the arrays calculated in the previous step.

All features of each _fruit slice_ will be concatenated at the end of the pipeline.

## Example
A simple example could look like this:
```python
# 3 dimensional time series dataset of 200 time series of length 100
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
fruit.add(fruits.sieving.PPV(quantile=0.5, constant=False))
fruit.add(fruits.sieving.MAX)

# cut a new fruit slice without the INC preparateur
fruit.cut()
fruit.add(iss.copy())
fruit.add(fruits.sieving.PPV(quantile=0, constant=True))
fruit.add(fruits.sieving.MIN)

# fit the fruit to the data and extract all features
fruit.fit(X_train)
X_train_features = fruit.transform(X_train)
```

## UCR-Experiments
Have a look at the [instructions](experiments/README.md) for more information on how to execute some experiments with __FRUITS__.

## Unit tests
There are a bunch of [tests](tests) for __FRUITS__ available to execute. To do this, enter the command
```
  $ python -m pytest tests
```
in a terminal/command line from the main directory of this repository.
