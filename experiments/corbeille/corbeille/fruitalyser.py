from timeit import default_timer as Timer
from typing import (Any, Callable, Literal, Optional, Protocol, Sequence,
                    TypeVar, overload)

import fruits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import RidgeClassifierCV

from .tools import split_index, transformation_string

_COLORS: list[tuple[int, int, int]] = [
    (0, 100, 173),
    (181, 22, 33),
    (87, 40, 98),
    (0, 48, 100),
    (140, 25, 44),
]

_TFitScoreSelf = TypeVar("_TFitScoreSelf", bound="FitScoreClassifier")
_TFitTransformSelf = TypeVar("_TFitTransformSelf", bound="FitTransform")


class FitScoreClassifier(Protocol):

    def score(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

    def fit(
        self: _TFitScoreSelf,
        X: np.ndarray,
        y: np.ndarray,
    ) -> _TFitScoreSelf:
        ...


class FitTransform(Protocol):

    def fit(
        self: _TFitTransformSelf,
        X: np.ndarray,
    ) -> _TFitTransformSelf:
        ...

    def transform(self, X: np.ndarray) -> Any:
        ...

    def fit_transform(self, X: np.ndarray) -> Any:
        ...


def get_color(i: int) -> tuple[float, float, float]:
    """Returns a color identified by the given index. Indices 0-4 return
    specific colors defined by the module, 5-19 yield colors from the
    colormap 'tab20b' in matplotlib and a higher index ``i>=20`` returns
    ``c[i%5]`` where ``c`` are the five first colors already mentioned.
    """
    if i < 5:
        return tuple(x/255 for x in _COLORS[i])
    elif i < 20:
        return cm.get_cmap("tab20b")(2+i-5)[:3]
    else:
        return tuple(x/255 for x in _COLORS[i % len(_COLORS)])


class _TransformationCallback(fruits.callback.AbstractCallback):
    """Callback that is needed to extract processed timeseries datasets
    within a fruit.
    """

    def __init__(
        self,
        prepared: bool = True,
        iterated: bool = True,
        sieved: bool = True,
    ) -> None:
        self._slc = -1
        self._iss = -1
        self._prepared = prepared
        if self._prepared:
            self._prepared_data: list[np.ndarray] = []
        self._iterated = iterated
        if self._iterated:
            self._iterated_sums: list[list[list[np.ndarray]]] = []
        self._sieved = sieved
        if self._sieved:
            self._sieved_data: list[np.ndarray] = []

    @property
    def prepared_data(self) -> list[np.ndarray]:
        if not self._prepared:
            raise RuntimeError("Didn't save prepared data")
        return self._prepared_data

    @property
    def iterated_sums(self) -> list[list[list[np.ndarray]]]:
        if not self._iterated:
            raise RuntimeError("Didn't save iterated sums")
        return self._iterated_sums

    @property
    def sieved_data(self) -> list[np.ndarray]:
        if not self._sieved:
            raise RuntimeError("Didn't save sieved data")
        return self._sieved_data

    def on_next_slice(self) -> None:
        self._slc += 1
        self._iss = -1

    def on_next_iss(self) -> None:
        self._iss += 1

    def on_preparation_end(self, X: np.ndarray) -> None:
        if self._prepared:
            self._prepared_data.append(X)

    def on_iterated_sum(self, X: np.ndarray) -> None:
        if self._iterated:
            if len(self._iterated_sums) >= self._slc:
                self._iterated_sums.append([])
            if len(self._iterated_sums[self._slc]) >= self._iss:
                self._iterated_sums[self._slc].append([])
            self._iterated_sums[self._slc][self._iss].append(X)

    def on_sieving_end(self, X: np.ndarray) -> None:
        if self._sieved:
            self._sieved_data.append(X)


class Fruitalyser:
    """Class for analysing features transformed by a ``fruits.Fruit``.

    Args:
        fruit (fruits.Fruit): The fruit to analyze.
        data (tuple of numpy arrays): Tuple of an already split dataset
            into training and testing data with the form
            ``(X_train, y_train, X_test, y_test)``.
    """

    def __init__(
        self,
        fruit: fruits.Fruit,
        data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.fruit = fruit
        self._extracted = False
        self._X_train, self._y_train, self._X_test, self._y_test = data
        self.callback: _TransformationCallback

    def transform_all(
        self,
        postprocess: Optional[FitTransform] = None,
        verbose: bool = False,
        save_prepared: bool = False,
        save_iterated: bool = False,
        save_sieved: bool = False,
    ) -> tuple[float, float]:
        """Transforms training and testing dataset.

        Args:
            postprocess (FitTransform, optional): An object of a class
                with methods ``fit`` and ``transform``. The features
                calculated by the fruit will be transformed with this
                transform.
            verbose (bool, optional): Increase verbosity of the
                transformation. This will print out timings on
                finished fit and transform steps. Defaults to False.

        Returns:
            float: Time for feature extraction in the training set.
            float: Time for feature extraction in the testing set.
        """
        self.callback = _TransformationCallback(
            save_prepared,
            save_iterated,
            save_sieved,
        )
        start = Timer()
        self.fruit.fit(self._X_train)
        if verbose:
            print(f"Fitting took {Timer() - start} s")
        start = Timer()
        self._X_train_feat = self.fruit.transform(self._X_train)
        if postprocess:
            self._X_train_feat = postprocess.fit_transform(self._X_train_feat)
        train_time = Timer() - start
        if verbose:
            print(f"Transforming training set took {train_time} s")
        start = Timer()
        self._X_test_feat = self.fruit.transform(
            self._X_test,
            callbacks=[self.callback],
        )
        if postprocess is not None:
            self._X_test_feat = postprocess.transform(self._X_test_feat)
        test_time = Timer() - start
        if verbose:
            print(f"Transforming testing set took {test_time} s")
        self._extracted = True
        return (train_time, test_time)

    def classify(
        self,
        indices: Optional[Sequence[int]] = None,
        classifier: Optional[FitScoreClassifier] = None,
        verbose: bool = False,
    ) -> float:
        """Starts the classification. Transforms the data first if
        necessary.

        Args:
            classifier (FitScoreClassifier, optional): Used classifier
                with a ``fit`` and ``score`` method. Defaults to
                ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))``.
            verbose (bool, optional): Increase verbosity of the
                transformation. This will print out timings on
                finished fit and transform steps. Defaults to False.

        Returns:
            float: The accuracy of the model on the test set.
        """
        if not self._extracted:
            raise RuntimeError(
                "Features not extracted, call ``transform_all`` first."
            )
        if classifier is None:
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

        if indices is not None:
            classifier.fit(self._X_train_feat[:, indices], self._y_train)
            self.test_score = float(
                classifier.score(self._X_test_feat[:, indices], self._y_test)
            )
        else:
            classifier.fit(self._X_train_feat, self._y_train)
            self.test_score = float(
                classifier.score(self._X_test_feat, self._y_test)
            )

        if verbose:
            print(f"Classification with {type(classifier)}")
            print(f"\t+ Accuracy on test set: {self.test_score}")
        return self.test_score

    def test_classifier(
        self,
        classifier: Callable[..., FitScoreClassifier],
        variable: str,
        test_cases: Sequence[Any],
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Tests a classifier for its accuracy on the calculated
        features. The classifier is initialized with different
        configurations and a 2D-plot is created for visualisation of the
        results.

        Args:
            classifier (Callable[..., FitScoreClassifier]): The
                (uninitialized) classifiers class to test.
            variable (str, optional): String of the name of the variable
                that is needed for initialization of the classifier.
            test_cases (sequence of values): Different values the
                'variable' argument of the classifier can accept. The
                classifier is initialized each time with a different
                value and the corresponding accuracies on the test set
                are plotted.
            kwargs: Keyword arguments passed to the classifier.

        Returns:
            A tuple of a matplotlib figure and axis.
        """
        accuracies = np.zeros(len(test_cases))
        for i, test_case in enumerate(test_cases):
            t = {variable: test_case}
            t = dict(kwargs, **t)
            clssfr = classifier(**t)
            clssfr.fit(self._X_train_feat, self._y_train)
            accuracies[i] = clssfr.score(self._X_test_feat, self._y_test)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_cases, accuracies, marker="x", label="test accuracy")
        ax.vlines(
            test_cases[np.argmax(accuracies)],
            min(accuracies),
            max(accuracies),
            color="red",
            linestyle="--",
            label="best result",
        )
        ax.set_title(f"Accuracy results using {classifier}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Accuracy")
        ax.legend(loc="upper right")
        return fig, ax

    @staticmethod
    def _plot(
        X: np.ndarray,
        y: np.ndarray,
        axis: Axes,
        mean: bool,
        bounds: bool,
        nseries: int,
        per_class: bool,
    ) -> None:
        classes = sorted(list(set(y)))
        if mean:
            for i in range(len(classes)):
                X_class = X[y == classes[i]]
                mean_X = X_class.mean(axis=0)
                color = get_color(i)
                axis.plot(mean_X, label=f"Class {i+1}", color=color)
                if bounds:
                    axis.fill_between(np.arange(len(mean_X)),
                                         X_class.max(axis=0),
                                         X_class.min(axis=0),
                                         color=color, alpha=.1)
            axis.legend(loc="best")
        else:
            if per_class:
                for i in range(len(classes)):
                    X_class = X[y == classes[i]]
                    color = get_color(i)
                    for j, ind in enumerate(range(nseries)):
                        if j == 0:
                            axis.plot(X_class[ind, :], label=f"Class {i+1}",
                                         color=color)
                        else:
                            axis.plot(X_class[ind, :], color=color)
                axis.legend(loc="best")
            else:
                for i in range(nseries):
                    axis.plot(X[i, :])

    @overload
    def plot(
        self,
        level: Literal["input", "prepared", "iterated sums"] = ...,
        index: Optional[int] = ...,
        dim: int = ...,
        mean: bool = ...,
        bounds: bool = ...,
        nseries: int = ...,
        per_class: bool = ...,
        axis: None = ...,
    ) -> tuple[Figure, Axes]:
        ...

    @overload
    def plot(
        self,
        level: Literal["input", "prepared", "iterated sums"] = ...,
        index: Optional[int] = ...,
        dim: int = ...,
        mean: bool = ...,
        bounds: bool = ...,
        nseries: int = ...,
        per_class: bool = ...,
        axis: Optional[Axes] = ...,
    ) -> None:
        ...

    def plot(
        self,
        level: Literal["input", "prepared", "iterated sums"] = "input",
        index: Optional[int] = None,
        dim: int = 0,
        mean: bool = False,
        bounds: bool = False,
        nseries: int = 1,
        per_class: bool = True,
        axis: Optional[Axes] = None,
    ) -> Optional[tuple[Figure, Axes]]:
        """Plots one dimension of the data for the given level.

        Args:
            level (str, optional): Stage of the transformation to plot
                the data from. Possible levels are
                ``['input', 'prepared', 'iterated sums']``.
                Defaults to plotting the input data (from the test set).
            index (int, optional): The index of the preparateur or word
                in the fruit counting over all slicees.
            dim (int, optional): Dimension of each time series to plot.
                Defaults to 0.
            mean (bool, optional): If ``True``, plots the mean of all
                time series. This would ignore ``nseries`` and
                ``per_class`` arguments. Defaults to False.
            bounds (bool, optional): Colors the area around the mean up
                to the maximum and down to the minimum iterated over the
                x axis. Defaults to False
            nseries (int, optional): Plots the given number of time
                series from the dataset. Uses the indices
                ``[0, ..., nseries-1]``. Defaults to 1.
            per_class (bool, optional): If ``True``, plots ``nseries``
                time series of each class in the dataset. Defaults to
                True.
            axis (Axes, optional): Matplotlib axis to plot the data
                on. If None is given, a seperate figure and axis
                will be created and returned.

        Returns:
            Tuple of a matplotlib figure and axes holding the inserted
            plot(s) or None if ``axis`` is provided.
        """
        fig, ax = (None, axis) if axis is not None else (
            plt.subplots(1, 1, figsize=(10, 5))
        )
        if level == "input":
            Fruitalyser._plot(
                self._X_test[:, dim, :],
                self._y_test,
                axis=ax,
                mean=mean,
                bounds=bounds,
                nseries=nseries,
                per_class=per_class,
            )
        elif level == "prepared":
            if index is None:
                index = 0
            Fruitalyser._plot(
                self.callback.prepared_data[index][:, dim, :],
                self._y_test,
                axis=ax,
                mean=mean,
                bounds=bounds,
                nseries=nseries,
                per_class=per_class,
            )
        elif level == "iterated sums":
            if index is None:
                index = 0
            indices = split_index(
                self.fruit,
                index=index,
                level="iterated sums",
            )
            el = 0
            iss = self.fruit[indices[0]].get_iss()[indices[1]]
            if (iss.mode == fruits.ISSMode.EXTENDED):
                el = indices[3] - (
                    len(iss.words[indices[2]])
                    - iss._cache_plan.unique_el_depth(indices[2])
                )
            Fruitalyser._plot(
                self.callback.iterated_sums[indices[0]][indices[1]][
                    indices[2]
                ][:, el, :],
                self._y_test,
                axis=ax,
                mean=mean,
                bounds=bounds,
                nseries=nseries,
                per_class=per_class,
            )
        else:
            raise ValueError(f"Unknown level: {level}")

        if level == "input":
            ax.set_title("Input Data")
        else:
            ax.set_title(transformation_string(
                self.fruit,
                index if index is not None else 0,
                level=level,
            ))

        if fig is not None:
            return fig, ax
        return None

    def features(
        self,
        indices: Optional[Sequence[int]] = None,
        source: Optional[Literal["train", "test", "all"]] = "all",
    ) -> pd.DataFrame:
        """Returns a ``pandas.DataFrame`` object with all features
        matching the following specifications.
        This method can only be used if ``self.classify()`` was called
        before.
        """
        if indices is None:
            indices = range(self.fruit.nfeatures())
        if source == "train":
            features = self._X_train_feat
        elif source == "test":
            features = self._X_test_feat
        elif source == "all":
            features = np.r_[self._X_train_feat, self._X_test_feat]
        else:
            raise ValueError(f"Unknown Value for option 'source': {source}")
        feat_table = np.empty(
            (len(indices), features.shape[0]),
            dtype=np.float64,
        )
        column_names = []
        for i, index in enumerate(indices):
            feat_table[i] = features[:, index]
            column_names.append(transformation_string(self.fruit, index))
        feats = pd.DataFrame(feat_table.T, columns=column_names)
        return feats

    def plot_features(
        self,
        indices: Optional[Sequence[int]] = None,
        source: Optional[Literal["train", "test", "all"]] = "all",
    ) -> sns.PairGrid:
        """Plots the features of the watched ``fruits.FruitSlice``
        object. The ``seaborn.pairplot`` function is used to create
        a plot based on a ``pandas.DataFrame`` consisting of selected
        features.

        Args:
            indices (Sequence of int): List of features indices in the
                fruit.

        Returns:
            seaborn.PairGrid: A PairGrid plot from the package seaborn.
        """
        feats = self.features(indices, source=source)
        if source == "train":
            feats["Class"] = self._y_train
        elif source == "test":
            feats["Class"] = self._y_test
        elif source == "all":
            feats["Class"] = np.r_[self._y_train, self._y_test]
        else:
            raise ValueError(f"Unknown Value for option 'source': {source}")
        pp = sns.pairplot(
            feats,
            hue="Class",
            diag_kind="hist",
            markers="+",
            palette=[
                get_color(i) for i in range(feats["Class"].value_counts().size)
            ],
        )
        pp.fig.suptitle(f"Features", y=1.01)
        return pp
