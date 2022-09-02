import os
import time
from collections.abc import Sequence
from timeit import default_timer as Timer
from typing import Optional, Union

import fruits
import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import load
from .fruitalyser import FitScoreClassifier


class Fruitifier:
    """A pipeline that connects feature extraction of a ``fruits.Fruit``
    with a classifier from sklearn for a classification of univariate or
    multivariate time series data.

    Args:
        additional_columns (str, optional): Path to a .csv file
            containing a table with a first column called 'Dataset'.
            All additional columns in that table will be appended to the
            resulting table of this experiment.
        verbose (bool, optional): Increase verbosity of the
            classification by printing current progress to the console.
        classifier (FitScoreClassifier, optional): Classifier with a fit
            and score method. Defaults to the sklearn classifier
            ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`` with a
            prior standardization of the features.
        comet_experiment (comet_ml.Experiment, optional): An experiment
            object from the package ``comet_ml`` that is used for saving
            the results from the classification.
    """

    output_header_names = [
        "Dataset",
        "FRUITS Time",
        "FRUITS Acc",
    ]

    def __init__(
        self,
        classifier: Optional[FitScoreClassifier] = None,
        additional_columns: Optional[str] = None,
        comet_experiment: Optional[Experiment] = None,
    ) -> None:
        self._results = []
        self._add_columns = (
            pd.read_csv(additional_columns) if additional_columns is not None
            else None
        )
        # dictionary path: ([datasets in path], univariate?)
        self._datasets: dict[str, tuple[list[str], bool]] = {}
        self._comet_exp = comet_experiment
        if classifier is None:
            self._classifier = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("nantonum", FunctionTransformer(np.nan_to_num)),
                ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
            ])
        else:
            self._classifier = classifier

    def append_data(
        self,
        path: str,
        names: Optional[Union[Sequence[str], str]] = None,
        univariate: bool = True,
    ) -> None:
        """Finds time series datasets in the specified directory for
        later usage.

        Args:
            path (str): Path of a directory that contains folders
                structured like the datasets you get from
                timeseriesclassification.com.
            names (str or sequence of str): List of dataset names. Only
                the datasets in this list will be collected. Defaults to
                all datasets in the given paths. If ``names`` is a
                single string, it will be treated as a path to a .txt
                file where all dataset names are listed and seperated by
                a newline character. Defaults to None.
            univariate (bool, optional): Whether the data in ``path`` is
                assumed to be univariate or multivariate. Defaults to
                ``True``.
        """
        if not path.endswith("/"):
            path += "/"
        self._datasets[path] = ([], univariate)
        if isinstance(names, str):
            with open(names, "r") as file:
                names = file.read().split("\n")
        for folder in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, folder)):
                if names is None or folder in names:
                    self._datasets[path][0].append(folder)
        if len(self._datasets[path][0]) == 0:
            del self._datasets[path]

    def _build_dataframe(self) -> pd.DataFrame:
        columns = self.output_header_names
        if self._add_columns is not None:
            columns += self._add_columns.columns
        df = pd.DataFrame(
            self._results,
            columns=columns,
        )
        return df

    def classify(
        self,
        fruit: Optional[fruits.Fruit] = None,
        cache_results: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Classifies all datasets added earlier and summarizes the
        results in a pandas DataFrame.

        Args:
            fruit (fruits.Fruit, optional): Fruit transform for feature
                extraction. Defaults to a fruit built with
                ``fruits.build`` for every dataset.
            fit_sample_size (float or int, optional): The option
                ``sample_size`` supplied to the fit method of the fruit.
                Defaults to 1.
            cache_results (str, optional): If a filename is supplied,
                the method will save results at the end of the
                classification for each dataset to a csv file with that
                name.
        """
        if cache_results is not None and cache_results.endswith(".csv"):
            cache_results = cache_results[:-4]
        self._results = []

        if self._comet_exp is not None:
            self._comet_exp.log_dataset_info(
                name=",".join([ds for path in self._datasets
                               for ds in self._datasets[path][0]])
            )
            self._comet_exp.log_text(
                fruit.summary() if fruit is not None
                else "Building a Fruit for each dataset..."
            )

        if verbose:
            print(f"Starting Classification")

        i = 0

        for path in self._datasets:

            univariate = self._datasets[path][1]

            for dataset in self._datasets[path][0]:

                results: list[Union[str, float]] = [dataset]

                if self._comet_exp is not None:
                    self._comet_exp.set_step(i)

                X_train, y_train, X_test, y_test = load(
                    os.path.join(path, dataset),
                    univariate=univariate,
                )
                X_train = np.nan_to_num(X_train)
                X_test = np.nan_to_num(X_test)

                fruit_ = fruit if fruit is not None else fruits.build(X_train)

                start = Timer()
                fruit_.fit(X_train)
                X_train_feat = fruit_.transform(X_train)
                X_test_feat = fruit_.transform(X_test)
                results.append(Timer() - start)

                self._classifier.fit(X_train_feat, y_train)
                results.append(self._classifier.score(X_test_feat, y_test))

                if self._add_columns is not None:
                    df = self._add_columns[
                        self._add_columns["dataset"] == dataset
                    ]
                    for c in self._add_columns.columns:
                        results.append(df[c].iloc[0])

                self._results.append(results)
                if verbose:
                    print(".", end="", flush=True)
                if cache_results is not None:
                    df = self._build_dataframe()
                    df.to_csv(f"{cache_results}.csv")
                    if self._comet_exp is not None:
                        self._comet_exp.log_table("results.csv", df)
                i += 1

        if self._comet_exp is not None:
            df = self._build_dataframe()
            self._comet_exp.log_table("results.csv", df)
            self._comet_exp.log_text(df.to_markdown(
                index=False,
                numalign="center",
                stralign="center",
            ))
            self._comet_exp.log_html(df.to_html(
                index=False,
                justify="center",
            ))

        if verbose:
            print(f"\nDone")

    def save_csv(self, filename: str) -> None:
        """Outputs all results of classified datasets to the file(s).

        Args:
            filename (str): If the given file name already exists, a
                timestamp is appended before saving.
        """
        if filename.endswith(".csv"):
            filename = filename[:-4]
        if os.path.exists(filename + ".csv"):
            filename += "_" + time.strftime("%Y-%m-%d-%H%M%S")

        df = self._build_dataframe()
        df.to_csv(filename + ".csv", index=False)
