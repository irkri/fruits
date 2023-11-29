# Fruits Extension: Corbeille

The small extension package [corbeille](/experiments/corbeille/) implements classes and methods
that help with classification experiments using ``fruits`` and their analysis.

## Installation

Dependecies for ``corbeille`` are listed in the corresponding
[pyproject.toml](/experiments/corbeille/pyproject.toml).

You can install these additional dependencies as well as the package using poetry. Execute the
following command within the root folder of this repository.

    $ poetry install --extras corbeille

Alternatively, directly use

    >>> python -m pip install .

in the [directory of corbeille](/experiments/corbeille) within a Python environment that already
has ``fruits`` installed.

## Execution

There are predefined ``fruits`` configurations available for testing in the
[experiments](/experiments) folder. A simple code to execute the experiments on data from
[timeseriesclassification.com](https://timeseriesclassification.com) (readable with numpy as .txt
format, see the module [corbeille.data](/experiments/corbeille/corbeille/data.py)) could be:

```python
import corbeille
from fruit_general import fruit

data = corbeille.data.load("path/to/your/data")

time, acc = corbeille.fruitify(data, fruit)
print(
    f"The fruit {fruit.name!r} needed {time:.2f} seconds to get an"
    f" accuracy of {100*acc:.2f} %."
)
```

The notebook [datasets.ipynb](/experiments/datasets.ipynb) contains experiments on datasets from
the UCR archive. Some of these datasets exhibit strange behavior when using FRUITS for feature
extraction.
