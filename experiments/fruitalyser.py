"""This python module is an appendix to the package FRUITS.

It contains the class Fruitalyser, which allows the user to take a
look at the calculations 'under the hood' of a :meth:`fruits.Fruit`
object when extracting features of a multidimensional time series
dataset.
"""

from timeit import default_timer as Timer

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from context import fruits

_CLASSIFIERS = [
    (RidgeClassifierCV, {"alphas": np.logspace(-3, 3, 10),
                         "normalize": True}),
    (SVC, {"kernel": "poly",
           "gamma": "auto",
           "degree": 1},
        "degree",
        list(range(1, 6))),
    (KNeighborsClassifier, {"n_neighbors": 1},
        "n_neighbors",
        list(range(1, 21))),
    (SGDClassifier, {"penalty": "l1",
                     "alpha": 0.0001,
                     "max_iter": 10_000},
        "max_iter",
        [100, 1000, 10_000, 50_000, 100_000]),
]

_COLORS = [
    (0, 100, 173),
    (181, 22, 33),
    (87, 40, 98),
    (0, 48, 100),
    (140, 25, 44),
]

def _get_color(i: int):
    if i < 5:
        return tuple(x/255 for x in _COLORS[i])
    elif i < 20:
        return mpl.cm.get_cmap("tab20b").colors[2:][i-5]
    else:
        return [x/255 for x in _COLORS[i % len(_COLORS)]]


class TransformationCallback(fruits.callback.AbstractCallback):
    def __init__(self, branch: int = 0):
        self._branch = branch
        self._current_branch = -1
        self.prepared_data = None
        self.iterated_sums = []
        self.sieved_data = None

    def on_next_branch(self):
        self._current_branch += 1

    def on_preparation_end(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self.prepared_data = X

    def on_iterated_sum(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self.iterated_sums.append(X)

    def on_sieving_end(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self.sieved_data = X


class Fruitalyser:
    """Class for analysing results for a fruits.Fruit object and its
    classification pipeline.

    :param fruit: A fruits configuration for feature
        extraction
    :type fruit: fruits.Fruit
    :param data: Tuple of a (already split) classification dataset
        ``[X_train, y_train, X_test, y_test]``. This format is the same
        you get from calling the module method
        ``fruitalyser.load_dataset``.
    :type data: tuple
    """
    def __init__(self,
                 fruit: fruits.Fruit,
                 data: tuple):
        self.fruit = fruit
        self._extracted = None
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.X = self.X_test
        self.y = self.y_test

    def classify(self,
                 classifier=None,
                 scaler=None,
                 watch_branch: int = 0,
                 test_set: bool = True):
        """Classifies the specified data by first extracting the
        features of the time series using the fruits.Fruit object.
        
        :param classifier: Used classifier, defaults to
            ```RidgeClassifierCV(alphas=np.logspace(-3,3,10), 
                                 normalize=True)```
        :type classifier: Classifier from the package sklearn with
            a fit and score method., optional
        :param scaler: Used scaler to scale the calculated features.,
            defaults to None
        :type scaler: Scaler from the package sklearn with a fit and
            transform method.
        :param watch_branch: The incremental steps (prepared data, 
            iterated sums) of a fruits.Fruit object can be only saved
            for one branch in this object. The index of this branch
            is specified with this argument.
            All future calls of other methods on this Fruitalyser
            object will depend on this option., defaults to 0
        :type watch_branch: int, optional
        :param test_set: If True, the results from the
            transformation of the test set will be used., defaults to
            True
        :type test_set: bool, optional
        """
        if self._extracted is None or self._extracted != watch_branch:
            self.callback = TransformationCallback(watch_branch)
            watched_branch = self.fruit.branches()[watch_branch]
            start = Timer()
            self.fruit.fit(self.X_train)
            print(f"Fitting took {Timer() - start} s")
            if test_set:
                self.X = self.X_test
                self.y = self.y_test
            else:
                self.X = self.X_train
                self.y = self.y_train
            start = Timer()
            if test_set:
                self.X_train_feat = self.fruit.transform(self.X_train)
            else:
                self.X_train_feat = self.fruit.transform(self.X_train,
                                        callbacks=[self.callback])
            print(f"Transforming training set took {Timer() - start} s")
            start = Timer()
            if test_set:
                self.X_test_feat = self.fruit.transform(self.X_test,
                                        callbacks=[self.callback])
            else:
                self.X_test_feat = self.fruit.transform(self.X_test)
            print(f"Transforming testing set took {Timer() - start} s")
            self._extracted = watch_branch

            self.preparateurs = watched_branch.get_preparateurs()
            self.words = watched_branch.get_words()
            self.sieves = watched_branch.get_sieves()
            self.nfeatures = sum([sieve.nfeatures() for sieve in self.sieves])
            self.nbranchfeatures = watched_branch.nfeatures()
        else:
            print("Features already extracted.")

        if scaler is not None:
            scaler.fit(self.X_train_feat)
            self.X_train_feat = scaler.transform(self.X_train_feat)
            self.X_test_feat = scaler.transform(self.X_test_feat)
        if classifier is None:
            classifier = RidgeClassifierCV(alphas=np.logspace(-3,3,10),
                                           normalize=True)
        classifier.fit(self.X_train_feat, self.y_train)
        self.test_score = classifier.score(self.X_test_feat,
                                           self.y_test)

        print(f"Classification with {type(classifier)}")
        print(f"\t+ Accuracy on test set: {self.test_score}")

    def test_classifier(self,
                        classifier,
                        variable: str = None,
                        test_cases: list = None,
                        **kwargs):
        """Tests a classifier for its accuracy on the features of the
        input data. The classifier is initialized with different
        configurations and a 2D-plot is created for visualisation of the
        results.
        
        :param classifier: Classifier to test.
        :type classifier: Class of a classifier from the package sklearn
            with a fit and score method.
        :param variable: String of the name of the variable that is
            needed for initialization of the classifier.
        :type variable: str
        :param test_cases: Different values the 'variable' argument of
            the classifier can accept. The classifier is initialized
            with all of these values and the results (accuracy on test
            set) are plottet.
        :type test_cases: list
        :param kwargs: Keyword arguments passed to the classifier.
        :type kwargs: unfold dict
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s).
        :rtype: tuple
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
        ax.vlines(test_cases[np.argmax(accuracies)],
                  min(accuracies),
                  max(accuracies),
                  color="red",
                  linestyle="--",
                  label="best result")
        ax.set_title(f"Accuracy results using {classifier}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Accuracy")
        ax.legend(loc="upper right")
        return fig, ax

    def classifier_battery(self):
        """Tests a bunch of classifiers for the already calculated
        features. Returns a list of (figure, axis) tuples from
        ``self.test_classifier()``.
        This method can only be used if ``self.classify()`` was called
        before.
        """
        figax = []
        for clssfr in _CLASSIFIERS:
            classifier = clssfr[0](**(clssfr[1]))
            classifier.fit(self.X_train_feat, self.y_train)
            score = classifier.score(self.X_test_feat, self.y_test)
            print(f"Classifier: {str(classifier)}\n\t-> Accuracy: {score}")
            if len(clssfr) > 2:
                figax.append(self.test_classifier(clssfr[0],
                                                  variable=clssfr[2],
                                                  test_cases=clssfr[3],
                                                  **(clssfr[1])))
        return figax

    def print_fruit(self):
        """Prints a summary of the fruits.Fruit object this Fruitalyser
        was initialized with.
        """
        print(self.fruit.summary())

    def _get_plot_template(self,
                           rows: int = 1,
                           cols: int = 1,
                           **kwargs) -> tuple:
        # returns a plotting template as a tuple (figure, axis)
        fig, axs = plt.subplots(rows, cols, figsize=(10*cols, 5*rows),
                                **kwargs)
        return fig, axs

    def _plot(self, X, y, on_axis,
              mean: bool,
              bounds: bool,
              nseries: int,
              per_class: bool) -> tuple:
        classes = sorted(list(set(y)))
        if mean:
            for i in range(len(classes)):
                X_class = X[y == classes[i]]
                mean_X = X_class.mean(axis=0)
                color = _get_color(i)
                on_axis.plot(mean_X, label=f"Class {i+1}", color=color)
                if bounds:
                    on_axis.fill_between(np.arange(len(mean_X)),
                                         X_class.max(axis=0),
                                         X_class.min(axis=0),
                                         color=color, alpha=.1)
            on_axis.legend(loc="best")
        else:
            if per_class:
                for i in range(len(classes)):
                    X_class = X[y == classes[i]]
                    indices = list(range(nseries))
                    color = _get_color(i)
                    for j, ind in enumerate(indices):
                        if j == 0:
                            on_axis.plot(X_class[ind, :], label=f"Class {i+1}",
                                         color=color)
                        else:
                            on_axis.plot(X_class[ind, :], color=color)
                on_axis.legend(loc="best")
            else:
                indices = list(range(nseries))
                for i in indices:
                    on_axis.plot(X[i, :])

    def plot_input_data(self,
                        dim: int = 0,
                        mean: bool = False,
                        bounds: bool = False,
                        nseries: int = 1,
                        per_class: bool = True,
                        on_axis = None) -> tuple:
        """Plots the unchanged input data.
        
        :param dim: Dimension of each time series to plot.,
            defaults to 0
        :type dim: int, optional
        :param mean: If ``True``, plots the mean of all time series.
            This would ignore ``nseries`` and ``per_class`` arguments.,
            defaults to False
        :type mean: bool, optional
        :param bounds: Colors the area around the mean up to the
            maximum and down to the minimum iterated over the x axis.,
            defaults to False
        :type bounds: bool, optional
        :param nseries: Plots the given number of time series from the
            dataset. Uses the indices ``[0, ..., nseries-1]``. Raises a
            ``ValueError`` if not that many time series exist.,
            defaults to 1
        :type nseries: int, optional
        :param per_class: If ``True``, plots ``nseries`` time series
            of each class in the dataset., defaults to True
        :type per_class: bool, optional
        :param on_axis: Matplotlib axis to plot the data on. If ``None``
            is given, a seperate figure and axis will be created.
        :type on_axis: Axis from matplotlib
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s) or None if ``on_axis`` is provided.
        :rtype: tuple
        """
        if on_axis is None:
            fig, on_axis = self._get_plot_template(1, 1)
        self._plot(self.X[:, dim, :], self.y, on_axis, mean, bounds, nseries,
                   per_class)
        if on_axis is None:
            return fig, on_axis

    def plot_prepared_data(self,
                           dim: int = 0,
                           mean: bool = False,
                           bounds: bool = False,
                           nseries: int = 1,
                           per_class: bool = True,
                           on_axis = None) -> tuple:
        """Plots specified dimension of the prepared data.
        This method can only be used if ``self.classify()`` was
        called before.
        
        :param dim: Dimension of each time series to plot.,
            defaults to 0
        :type dim: int, optional
        :param mean: If ``True``, plots the mean of all time series.
            This would ignore ``nseries`` and ``per_class`` arguments.,
            defaults to False
        :type mean: bool, optional
        :param bounds: Colors the area around the mean up to the
            maximum and down to the minimum iterated over the x axis.,
            defaults to False
        :type bounds: bool, optional
        :param nseries: Plots the given number of time series from the
            dataset. Uses the indices ``[0, ..., nseries-1]``. Raises a
            ``ValueError`` if not that many time series exist.,
            defaults to 1
        :type nseries: int, optional
        :param per_class: If ``True``, plots ``nseries`` time series
            of each class in the dataset., defaults to True
        :type per_class: bool, optional
        :param on_axis: Matplotlib axis to plot the data on. If ``None``
            is given, a seperate figure and axis will be created.
        :type on_axis: Axis from matplotlib
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s) or None if ``on_axis`` is provided.
        :rtype: tuple
        """
        if on_axis is None:
            fig, on_axis = self._get_plot_template(1, 1)
        self._plot(self.callback.prepared_data[:, dim, :], self.y, on_axis,
                   mean, bounds, nseries, per_class)
        if on_axis is None:
            return fig, on_axis

    def plot_iterated_sums(self,
                           word_indices: list = None,
                           mean: bool = False,
                           bounds: bool = False,
                           nseries: int = 1,
                           per_class: bool = True,
                           on_axes = None) -> tuple:
        """Plots the iterated sums calculated in the specified
        fruits.Fruit object with ``fruitalyser.msplot()``.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param word_indices: Indices of the words that are used for
            plotting. If ``None``, all words are used., defaults to None
        :type word_indices: list, optional
        :param dim: Dimension of each time series to plot.,
            defaults to 0
        :type dim: int, optional
        :param mean: If ``True``, plots the mean of all time series.
            This would ignore ``nseries`` and ``per_class`` arguments.,
            defaults to False
        :type mean: bool, optional
        :param bounds: Colors the area around the mean up to the
            maximum and down to the minimum iterated over the x axis.,
            defaults to False
        :type bounds: bool, optional
        :param nseries: Plots the given number of time series from the
            dataset. Uses the indices ``[0, ..., nseries-1]``. Raises a
            ``ValueError`` if not that many time series exist.,
            defaults to 1
        :type nseries: int, optional
        :param per_class: If ``True``, plots ``nseries`` time series
            of each class in the dataset., defaults to True
        :type per_class: bool, optional
        :param on_axes: List of matplotlib axes to plot the data on.
            The list has to have the at least the same length as
            ``word_indices``. If ``None`` is given, a seperate figure
            and axes will be created.
        :type on_axes: List of axes from matplotlib
        :returns: Tuple (figure, axes) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s) or None if ``on_axes`` is provided.
        :rtype: tuple
        """
        if word_indices is None:
            word_indices = list(range(len(self.words)))
        if on_axes is None:
            fig, on_axes = self._get_plot_template(len(word_indices), 1,
                                                   sharex=True)
            if len(word_indices) == 1:
                on_axes = [on_axes]
        for i, index in enumerate(word_indices):
            self._plot(self.callback.iterated_sums[index][:, 0, :], self.y,
                       on_axes[i], mean, bounds, nseries, per_class)
            on_axes[i].set_title(str(self.words[index]))
        if on_axes is None:
            return fig, on_axes

    def plot_features(self,
                      sieve_index: int,
                      word_index: int) -> sns.PairGrid:
        """Plots the features of the watched ``fruits.FruitBranch``
        object. The ``seaborn.pairplot`` function is used to create
        a plot based on a ``pandas.DataFrame`` consisting of the
        following data.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index or indices of the sieve(s) that are
            used.
        :type sieve_index: int or list of int
        :param word_index: Index or indices of the word(s) that are
            used.
        :type word_index: int or list of int
        :returns: Pairplot of the specified features.
        :rtype: seaborn.PairGrid
        """
        feats = self.get_feature_dataframe(sieve_index, word_index)
        pp = sns.pairplot(feats,
                          hue="Class",
                          diag_kind="hist",
                          markers="+",
                          palette=[_get_color(i)
                                   for i in range(len(set(self.y)))])

        if isinstance(sieve_index, int):
            string = repr(self.sieves[self.sieve_index_to_sieve(sieve_index)])
            string = string.split(".")[-1]
            pp.fig.suptitle(f"Features of Sieve {string}", y=1.01)
        else:
            pp.fig.suptitle(f"Features of Word {self.words[word_index]}",
                            y=1.08)

        return pp

    def get_feature_dataframe(self,
                              sieve_index: int,
                              word_index: int) -> np.ndarray:
        """Returns a ``pandas.DataFrame`` object with all features
        matching the following specifications.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index or indices of the sieve(s) that are
            used.
        :type sieve_index: int or list of int
        :param word_index: Index or indices of the word(s) that are
            used.
        :type word_index: list or list of int
        :rtype: pandas.DataFrame
        """
        if isinstance(sieve_index, int) and isinstance(word_index, list):
            feat_table = np.array([self.callback.sieved_data[:,
                                    self.get_feature_index(sieve_index, i)]
                                   for i in word_index], dtype=np.float64)
            column_names = [str(self.words[i]) + ":" + str(i+1)
                            if len(str(self.words[i]))<8
                            else str(self.words[i])[:5] + "..." + ":" + str(i)
                            for i in word_index]
            feats = pd.DataFrame(feat_table.T, columns=column_names)
            feats["Class"] = self.y
            return feats

        elif isinstance(sieve_index, list) and isinstance(word_index, int):
            feat_table = np.array([self.callback.sieved_data[:,
                                    self.get_feature_index(i, word_index)]
                                   for i in sieve_index], dtype=np.float64)
            column_names = []
            for i in sieve_index:
                string = repr(self.sieves[self.sieve_index_to_sieve(i)])
                string = string.split(".")[-1]
                if len(string) < 8:
                    column_names.append(string + ":" + str(i))
                else:
                    column_names.append(string[:5] + "..." + ":" + str(i))
            feats = pd.DataFrame(feat_table.T,
                                 columns=column_names)
            feats["Class"] = self.y
            return feats

        else:
            raise ValueError("One of the two arguments has to be an integer, "
                             + "the other one a list.")

    def pca_correlation(self,
                        components: int) -> pd.DataFrame:
        """Returns a ``pandas.DataFrame`` object containing the
        correlation of the calculated features with the features given
        by a fitted ``sklearn.decomposition.PCA`` object.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param components: Number of components the PCA should
            calculate.
        :type components: int
        :rtype: pandas.DataFrame
        """
        pca = PCA(n_components=components)
        # standardize feature set
        pca.fit(self.callback.sieved_data)
        feature_pc_correlation = pd.DataFrame(pca.components_,
            columns=['Feat-'+str(i+1) for i in range(self.nbranchfeatures)],
            index=['PC-'+str(i+1) for i in range(components)])
        return feature_pc_correlation

    def rank_words_and_sieves(self, n: int, translate: bool = False) -> list:
        """Ranks the words and sieves by variance in the feature space
        (using PCA) in the watched FruitBranch.
        
        :param n: Number of objects to rank
        :type n: int
        :param translate: If True, the output will contain the names
            of the corresponding words and sieves. If False, the indices
            will be shown., defaults to False
        :type translate: bool, optional
        :returns: List of tuples ``(word, sieve)`` of length ``n``.
        :rtype: list
        """
        correlation = self.pca_correlation(n)
        sorted_feature_indices = [np.argmax(correlation.iloc[i])
                                  for i in range(n)]
        word_sieves_indices = [self.split_feature_index(feat)
                               for feat in sorted_feature_indices]
        if not translate:
            return word_sieves_indices
        words = [str(self.words[x[0]]) for x in word_sieves_indices]
        sieves = [str(self.sieves[self.sieve_index_to_sieve(x[1])])
                  for x in word_sieves_indices]
        return list(zip(words, sieves))

    def sieve_index_to_sieve(self, sieve_index: int) -> int:
        """Returns the name of the FeatureSieve that produces the
        feature at the given sieve index.
        
        :param sieve_index: Index of the sieved feature in the current
            branch.
        :type sieve_index: int
        :rtype: int
        """
        s = 0
        for i, n_i in enumerate([sieve.nfeatures() for sieve in self.sieves]):
            if sieve_index < s+n_i:
                return i
            s += n_i

    def split_feature_index(self, index: int) -> tuple:
        """For a given index for one of the calculated features from
        the ``fruits.Fruit`` object, this method returns a tuple
        containing the index of the word and the index of
        the FeatureSieve that led to this particular feature.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param index: Feature index
        :type index: int
        :rtype: tuple
        """
        word_index = index // self.nfeatures
        sieve_index =  index - self.nfeatures*word_index
        return (word_index, sieve_index)

    def get_feature_index(self, sieve_index: int, word_index: int) -> int:
        """Returns a feature index in the array of all concatenated
        features.
        This method can only be used if ``self.classify()`` was called
        before.

        :param sieve_index: Index of the corresponding sieve.
        :type sieve_index: int
        :param word_index: Index of the corresponding word.
        :type word_index: int
        :rtype: int
        """
        index = word_index * self.nfeatures
        return index + sieve_index
