import os
import time
from collections import Sequence
from timeit import default_timer as Timer
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split

import fruits

from .data import load_all
from .fruitalyser import FitScoreClassifier


def fruitify(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fruit: Union[fruits.Fruit,
                 Callable[[np.ndarray, np.ndarray], fruits.Fruit]],
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

    if callable(fruit):
        fruit = fruit(X_train, y_train)
    times, accs = [], []

    for _ in range(mean_over_n_runs):
        start = Timer()
        fruit.fit(X_train)
        X_train_feat = fruit.transform(X_train)
        X_test_feat = fruit.transform(X_test)
        times.append(Timer() - start)

        classifier.fit(X_train_feat, y_train)  # type: ignore
        accs.append(classifier.score(X_test_feat, y_test))  # type: ignore

    return float(np.mean(times)), float(np.mean(accs))


def fruitify_all(
    path: str,
    fruit: Union[fruits.Fruit,
                 Callable[[np.ndarray, np.ndarray], fruits.Fruit]],
    datasets: Optional[Sequence[str]] = None,
    univariate: bool = True,
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

    for data in load_all(path, univariate=univariate, datasets=datasets):
        timing, acc = fruitify(data[1:], fruit, classifier, mean_over_n_runs)
        results.append((data[0], acc, timing))
        pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Time"],
        ).to_csv(output_csv, index=False)

    return pd.DataFrame(results, columns=["Dataset", "Accuracy", "Time"])


def decide_which_fruit(
    choices: Sequence[Union[fruits.Fruit, tuple[fruits.Fruit, fruits.Fruit]]],
    n_splits: int = 1,
    validation_size: float = 0.2,
    classifier: Optional[FitScoreClassifier] = None,
    mean_over_n_runs: int = 1,
) -> Callable[[np.ndarray, np.ndarray], fruits.Fruit]:
    """Returns a function that decides which fruit out of the given set
    to choose for a dataset based on a number of experiments with
    validation training data.

    Args:
        choices (sequence of Fruit or of 2-tuples of Fruit): A sequence
            of fruits to choose from. If a tuple of two fruits is given,
            the choice is made based on experiments using the first
            fruit and the second one is returned afterwards. This way
            the choice can be made for an easy fruit and a more complex
            but structural similar one can be used for the overall
            classification.
        n_splits (int, optional): The number of validation splits to
            perform. Accuracies from these splits will be averaged
            afterwards. Defaults to 1.
        validation_size (float, optional): Size of the validation set.
            This set will be randomized ``n_splits`` times. Defaults to
            0.2.
        classifier (FitScoreClassifier): A classifier with a ``fit`` and
            ``score`` method. Defaults to a RidgeClassifierCV from the
            package sklearn with a prior standardization of the input
            features.
        mean_over_n_runs (int, optional): The classification for a
            single validation split is repeated the given number of
            times. Defaults to 1.
    """
    def choose(X: np.ndarray, y: np.ndarray) -> fruits.Fruit:
        choice_accuracies = []
        for choice in choices:
            accs = []
            for _ in range(n_splits):
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=validation_size, stratify=y,
                )
                choice_ = choice[0] if isinstance(choice, tuple) else choice
                acc, _ = fruitify(
                    (x_train, y_train, x_test, y_test),
                    choice_,
                    classifier=classifier,
                    mean_over_n_runs=mean_over_n_runs,
                )
                accs.append(acc)
            choice_accuracies.append(np.mean(accs))
        fruit = choices[np.argmax(choice_accuracies)]
        if isinstance(fruit, tuple):
            return fruit[1].deepcopy()
        return fruit.deepcopy()
    return choose
