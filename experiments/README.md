# Fruits Extension: Corbeille

The small extension package [corbeille](/experiments/corbeille/) implements classes and methods
that help with classification experiments using ``fruits`` and their analysis.

## Installation

Dependecies for ``corbeille`` are listed in the corresponding
[pyproject.toml](/experiments/corbeille/pyproject.toml).

You can install these additional dependencies as well as the package using poetry. Execute the
following command within the root folder of this repository.

    >>> poetry install --extras corbeille

Alternatively you can install ``corbeille`` with

    >>> python -m pip install .

in the [directory of corbeille](/experiments/corbeille) within a python environment that already
has ``fruits`` installed.

## Execution

There are predefined ``fruits`` configurations available for testing in the file
[configs.py](/experiments/configs.py). A simple code to execute the experiments on data from
[timeseriesclassification.com](https://timeseriesclassification.com) (readable with numpy as .txt
format, see the module [corbeille.data](/experiments/corbeille/corbeille/data.py)) could be:
```python
import corbeille
from configs import basket

data = corbeille.data.load("path/to/your/data")

for fruit in basket["apple"]:
    time, acc = corbeille.fruitify(data, fruit)
    print(
        f"The fruit {fruit.name!r} needed {time:.2f} seconds to get an"
        f" accuracy of {100*acc:.2f} %."
    )
```
