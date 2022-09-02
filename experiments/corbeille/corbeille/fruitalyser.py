from timeit import default_timer as Timer
from typing import (Any, Callable, Literal, Optional, Protocol, Sequence,
                    TypeVar)

import fruits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
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

    def __init__(self) -> None:
        self._current_branch = -1
        self.prepared_data: list[np.ndarray] = []
        self.iterated_sums: list[list[np.ndarray]] = []
        self.sieved_data: list[np.ndarray] = []

    def on_next_branch(self) -> None:
        self._current_branch += 1

    def on_preparation_end(self, X: np.ndarray) -> None:
        self.prepared_data.append(X)

    def on_iterated_sum(self, X: np.ndarray) -> None:
        if len(self.iterated_sums) >= self._current_branch:
            self.iterated_sums.append([])
        self.iterated_sums[self._current_branch].append(X)

    def on_sieving_end(self, X: np.ndarray) -> None:
        self.sieved_data.append(X)


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
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.X_train = np.nan_to_num(self.X_train)
        self.X_test = np.nan_to_num(self.X_test)
        self.X = self.X_test
        self.y = self.y_test
        self.callback: _TransformationCallback

    def transform_all(
        self,
        postprocess: Optional[FitTransform] = None,
        verbose: bool = False,
    ) -> None:
        """Transforms training and testing dataset.

        Args:
            postprocess (FitTransform, optional): An object of a class
                with methods ``fit`` and ``transform``. The features
                calculated by the fruit will be transformed with this
                transform.
            verbose (bool, optional): Increase verbosity of the
                transformation. This will print out timings on
                finished fit and transform steps. Defaults to False.
        """
        self.callback = _TransformationCallback()
        start = Timer()
        self.fruit.fit(self.X_train)
        if verbose:
            print(f"Fitting took {Timer() - start} s")
        start = Timer()
        self.X_train_feat = self.fruit.transform(self.X_train)
        if postprocess:
            self.X_train_feat = postprocess.fit_transform(self.X_train_feat)
        if verbose:
            print(f"Transforming training set took {Timer() - start} s")
        start = Timer()
        self.X_test_feat = self.fruit.transform(
            self.X_test,
            callbacks=[self.callback],
        )
        if postprocess is not None:
            self.X_test_feat = postprocess.transform(self.X_test_feat)
        if verbose:
            print(f"Transforming testing set took {Timer() - start} s")
        self._extracted = True

    def classify(
        self,
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
        classifier.fit(self.X_train_feat, self.y_train)
        self.test_score = float(
            classifier.score(self.X_test_feat, self.y_test)
        )

        if verbose:
            print(f"Classification with {type(classifier)}")
            print(f"\t+ Accuracy on test set: {self.test_score}")
        return self.test_score

    def test_classifier(
        self,
        classifier: Callable[..., FitScoreClassifier],
        variable: Optional[str] = None,
        test_cases: Optional[list] = None,
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
            test_cases (list): Different values the 'variable' argument
                of the classifier can accept. The classifier is
                initialized with all of these values and the results
                (accuracy on test set) are plottet.
            kwargs: Keyword arguments passed to the classifier.

        Returns:
            A tuple of figure and axis with the created plot(s).
        """
        if variable is None or test_cases is None:
            raise ValueError("Please specify a variable and a number of" +
                             "test cases")
        accuracies = np.zeros(len(test_cases))
        for i, test_case in enumerate(test_cases):
            t = {variable: test_case}
            t = dict(kwargs, **t)
            clssfr = classifier(**t)
            clssfr.fit(self.X_train_feat, self.y_train)
            accuracies[i] = clssfr.score(self.X_test_feat, self.y_test)
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

    def _plot(
        self,
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
                    indices = list(range(nseries))
                    color = get_color(i)
                    for j, ind in enumerate(indices):
                        if j == 0:
                            axis.plot(X_class[ind, :], label=f"Class {i+1}",
                                         color=color)
                        else:
                            axis.plot(X_class[ind, :], color=color)
                axis.legend(loc="best")
            else:
                indices = list(range(nseries))
                for i in indices:
                    axis.plot(X[i, :])

    def plot(
        self,
        level: Literal["input", "prepared", "iterated sums"] = "input",
        identifier: Optional[tuple[int, ...]] = None,
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
                Defaults to plotting the input data.
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
            self._plot(
                self.X[:, dim, :],
                self.y,
                axis=ax,
                mean=mean,
                bounds=bounds,
                nseries=nseries,
                per_class=per_class,
            )
        elif level == "prepared":
            if identifier is None:
                identifier = (0, )
            self._plot(
                self.callback.prepared_data[identifier[0]][:, dim, :],
                self.y,
                axis=ax,
                mean=mean,
                bounds=bounds,
                nseries=nseries,
                per_class=per_class,
            )
        elif level == "iterated sums":
            if identifier is None:
                identifier = (0, 0)
            self._plot(
                self.callback.iterated_sums[identifier[0]][identifier[0]][
                    :, dim, :
                ],
                self.y,
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
                identifier if identifier is not None else 0,
                level=level,
            ))

        if fig is not None:
            return fig, ax
        return None

    def get_feature_dataframe(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """Returns a ``pandas.DataFrame`` object with all features
        matching the following specifications.
        This method can only be used if ``self.classify()`` was called
        before.
        """
        if indices is None:
            indices = list(range(self.fruit.nfeatures()))
        feat_table = np.empty(
            (len(indices), self.X.shape[0]),
            dtype=np.float64,
        )
        column_names = []
        for i, index in enumerate(indices):
            sindex = split_index(self.fruit, index)
            findex = index
            for bindex in range(sindex[0]+1):
                findex -= self.fruit.branch(bindex).nfeatures()
            feat_table[i] = self.callback.sieved_data[sindex[0]][:, findex]
            column_names.append(transformation_string(self.fruit, sindex))
        feats = pd.DataFrame(feat_table.T, columns=column_names)
        return feats

    def plot_features(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> sns.PairGrid:
        """Plots the features of the watched ``fruits.FruitBranch``
        object. The ``seaborn.pairplot`` function is used to create
        a plot based on a ``pandas.DataFrame`` consisting of selected
        features.

        Args:
            indices (Sequence of int): List of features indices in the
                fruit.

        Returns:
            seaborn.PairGrid: A PairGrid plot from the package seaborn.
        """
        feats = self.get_feature_dataframe(indices)
        pp = sns.pairplot(
            feats,
            hue="Class",
            diag_kind="hist",
            markers="+",
            palette=[get_color(i) for i in range(len(set(self.y)))],
        )
        pp.fig.suptitle(f"Features", y=1.01)
        return pp

    def pca_correlation(
        self,
        components: int,
        indices: Optional[Sequence[int]] = None,
        translate: bool = True,
    ) -> pd.DataFrame:
        """Returns a ``pandas.DataFrame`` object containing the
        projected features in the principal axes calculated with
        ``sklearn.decomposition.PCA``.

        Args:
            components (int): Number of components for the PCA to
                calculate.
            indices (Sequence of int, optional): List of feature indices
                to use for the PCA. Defaults to all features.
            translate (bool, optional): Whether to translate the feature
                indices into readable string identifiers. Defaults to
                True.
        """
        pca = PCA(n_components=components)
        if indices is None:
            indices = list(range(self.fruit.nfeatures()))
        features = self.get_feature_dataframe(indices).to_numpy()
        pca.fit(features)
        if translate:
            columns = [transformation_string(self.fruit, i) for i in indices]
        else:
            columns = list(indices)
        feature_pc_correlation = pd.DataFrame(
            pca.components_,  # type: ignore
            columns=columns,
            index=[f"PC-{i+1}" for i in range(components)],
        )
        return feature_pc_correlation

    def rank_features(self, n: int, translate: bool = True) -> tuple[str, ...]:
        """Ranks features by variance using a PCA.

        Args:
            n (int): Number of features to rank.
            translate (bool, optional): Whether to translate the feature
                indices into readable string identifiers. Defaults to
                True.
        """
        correlation = self.pca_correlation(components=n, translate=translate)
        sorted_feature_indices = tuple(
            np.argmax(np.abs(correlation.iloc[i])) for i in range(n)
        )
        return tuple(
            str(correlation.columns[i]) for i in sorted_feature_indices
        )
