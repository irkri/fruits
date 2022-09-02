# Fruits Extension: Corbeille

The small extension package [corbeille](/experiments/corbeille/) implements classes and methods
that help with classification experiments using ``fruits`` and their analysis.

## Installation

Dependecies for ``corbeille`` are listed in the corresponding
[pyproject.toml](/experiments/corbeille/pyproject.toml).

You can install these additional dependencies as well as the package using poetry. Execute the following command within the root folder of this repository.

    >>> poetry install --extras corbeille

## Execution

There are predefined ``fruits`` configurations available for testing in the file
[configs.py](/experiments/configs.py). A simple code to execute the experiments on data from
[timeseriesclassification.com](https://timeseriesclassification.com) (readable with numpy as .txt
format, see the module [tsdata](/experiments/tsdata.py)) could be:
```python
from experiment import FRUITSExperiment
from configs import CONFIGS

pipeline = FRUITSExperiment()

pipeline.append_data("path/to/your/data")

for i, fruit in enumerate(CONFIGS):
    pipeline.classify(fruit)
    pipeline.produce_output(
        f"results_config_{i+1:02}",
        txt=True,
        csv=True,
    )
```
