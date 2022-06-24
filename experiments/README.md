# Experiment Instructions

## Dependencies

Extra dependecies may be needed for the execution of some of the here listed appendix modules for ``fruits``:

- ``matplotlib>=3.3.4``
- ``pandas>=1.2.5``
- ``scikit-learn>=0.24.2``
- ``seaborn>=0.11.1``
- ``scipy>=1.7.1``
- ``comet_ml>=3.13.1``
- ``networkx>=2.6.2``

Earlier versions of the named packages could work too, but aren't tested. You can install these
additional dependencies using poetry and the file [pyproject.toml](/pyproject.toml).
```
    $ poetry install -E experiment-dependencies
```

## Execution

There are predefined ``fruits`` configurations to test in the file [configs.py](/experiments/configs.py).
A simple code to execute the experiments on data from [timeseriesclassification.com](https://timeseriesclassification.com) (readable with numpy as .txt format, see the module [tsdata](/experiments/tsdata.py)) could be:
```python
from experiment import FRUITSExperiment
from configs import CONFIGS

pipeline = FRUITSExperiment()

pipeline.append_data("path/to/your/data")

for i, fruit in enumerate(CONFIGS):
    pipeline.classify(fruit)
    pipeline.produce_output("results_config_" + str(i+1).zfill(2),
                            txt=True, csv=True)
```
The docstrings provided in [experiment.py](/experiments/experiment.py), [fruitalyser.py](/experiments/fruitalyser.py), [tsdata.py](/experiments/tsdata.py) and [comparision.py](/experiments/comparision.py) should be enough to get a good understanding of what is possible to do with the defined classes.
