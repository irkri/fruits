import os
import time
from timeit import default_timer as Timer
from typing import Optional, Sequence

import fruits
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import load, load_all
from .fruitalyser import FitScoreClassifier


def fruitify(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fruit: Optional[fruits.Fruit] = None,
    classifier: Optional[FitScoreClassifier] = None,
    mean_over_n_runs: int = 1,
) -> tuple[float, float]:
    """Classifies the given time series dataset using the given fruit
    and classifier.

    Args:
        dataset (tuple of np.ndarray): A tuple of numpy array with shape
            ``(X_train, y_train, X_test, y_test)``.
        fruit (fruits.Fruit, optional): The Fruit used for feature
            extraction. Defaults to a Fruit built by fruits.
        classifier (FitScoreClassifier): A classifier with a ``fit`` and
            ``score`` method. Defaults to a RidgeClassifierCV from the
            package sklearn with a prior standardization of the input
            features.
        mean_over_n_runs (int, optional): The method repeats the
            classification a given number of times and returns the
            average time and accuracy. Defaults to 1.

    Returns:
        float: The time needed to extract the features with the
            specified Fruit.
        float: The accuracy result of the classification.
    """
    X_train, y_train, X_test, y_test = dataset
    if classifier is None:
        classifier = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("nantonum", FunctionTransformer(np.nan_to_num)),
            ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        ])  # type: ignore
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    fruit = fruit if fruit is not None else fruits.build(X_train)
    times, accs = [], []

    for _ in range(mean_over_n_runs):
        start = Timer()
        fruit.fit(X_train)
        X_train_feat = fruit.transform(X_train)
        X_test_feat = fruit.transform(X_test)
        times.append(Timer() - start)

        classifier.fit(X_train_feat, y_train)  # type: ignore
        accs.append(classifier.score(X_test_feat, y_test))  # type: ignore

    return np.mean(times), np.mean(accs)


def fruitify_all(
    path: str,
    univariate: bool = True,
    datasets: Optional[Sequence[str]] = None,
    fruit: Optional[fruits.Fruit] = None,
    classifier: Optional[FitScoreClassifier] = None,
    output_csv: Optional[str] = None,
    mean_over_n_runs: int = 1,
) -> pd.DataFrame:
    if output_csv is None:
        output_csv = "results_fruits"
    elif output_csv.endswith(".csv"):
        output_csv = output_csv[:-4]
    if os.path.exists(output_csv + ".csv"):
        output_csv += "_" + time.strftime("%Y-%m-%d-%H%M%S")
    output_csv = output_csv + ".csv"

    results: list[tuple[str, float, float]] = []

    for data in load_all(path, univariate, datasets):
        timing, acc = fruitify(data[1:], fruit, classifier, mean_over_n_runs)
        results.append((data[0], acc, timing))
        pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Time"],
        ).to_csv(output_csv, index=False)

    return pd.DataFrame(results, columns=["Dataset", "Accuracy", "Time"])
