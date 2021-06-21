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
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV

from context import fruits

def msplot(X: np.ndarray,
           y: np.ndarray,
           above_error: bool = True,
           below_error: bool = True,
           figure_height: int = 5,
           figure_width: int = 10,
           use_axes: Axes = None) -> tuple:
    """Generates a plot of a time series dataset that is already
    classified. A line for the mean of all time series in each class
    is drawn.
    
    :param X: (onedimensional) time series dataset,
    :type X: np.ndarray
    :param y: target values (i.e. classes) matching the order of the
        time series in ``X``
    :type y: np.ndarray
    :param above_error: If set to `True`, the space above the plotted
        line will be filled up to the standard deviation of all time
        series at a point., defaults to True
    :type above_error: bool, optional
    :param below_error: Same as 'above_error' except it's the space
        below the mean line that is filled., defaults to True
    :type below_error: bool, optional
    :param figure_height: Height of the plot., defaults to 5
    :type figure_height: int, optional
    :param figure_width: Width of the plot., defaults to 10
    :type figure_width: int, optional
    :param use_axes: If an axes is supplied, then the plot will be
        inserted and the axes is returned. Else, a new axes is created
        first., defaults to None
    :type use_axes: bool
    :returns: Axes with inserted plot and given size
    :rtype: Axes
    """
    classes = sorted(list(set(y)))
    if use_axes is None:
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    else:
        ax = use_axes
    for i in range(len(classes)):
        X_class = X[y==classes[i]]
        mean = X_class.mean(axis=0)
        std = X_class.std(axis=0)
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(mean, label=f"class {i+1}", color=color);
        above = mean
        below = mean
        if above_error:
            above = mean + std
        if below_error:
            below = mean - std
        ax.fill_between(np.arange(len(mean)),
                        below, above, color=color, alpha=.1)
    ax.legend(loc="upper right")
    return ax

def load_dataset(path: str) -> tuple:
    """Returns a time series dataset that is formatted as a .txt file
    and readable with numpy.
    
    :param path: Path to the dataset. The path has to point to a
        folder 'name' with two files:
        'name_TEST.txt' and 'name_TRAIN.txt' where 'name' is the name of
        the dataset.
    :type path: str
    :returns: Tuple (X_train, y_train, X_test, y_test) where X_train
        and X_test are 3-dimensional numpy arrays
    :rtype: tuple
    """
    dataset = path.split("/")[-1]
    train_raw = np.loadtxt(f"{path}/{dataset}_TRAIN.txt")
    test_raw = np.loadtxt(f"{path}/{dataset}_TEST.txt")
    X_train = train_raw[:, 1:]
    y_train = train_raw[:, 0].astype(np.int32)
    X_test = test_raw[:, 1:]
    y_test =  test_raw[:, 0].astype(np.int32)

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    return X_train, y_train, X_test, y_test

class TransformationCallback(fruits.callback.AbstractCallback):
    def __init__(self, branch: int = 0):
        self._branch = branch
        self._current_branch = -1
        self._prepared_data = None
        self._iterated_sums = []
        self._sieved_data = None

    def on_next_branch(self):
        self._current_branch += 1

    def on_preparation_end(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self._prepared_data = X

    def on_iterated_sum(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self._iterated_sums.append(X)

    def on_sieving_end(self, X: np.ndarray):
        if self._current_branch == self._branch:
            self._sieved_data = X


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
        self.X_train , self.y_train, self.X_test, self.y_test = data

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
            transform mehtod.
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
            self._callback = TransformationCallback(watch_branch)
            watched_branch = self.fruit.branches()[watch_branch]
            start = Timer()
            self.fruit.fit(self.X_train)
            print(f"Fitting took {Timer() - start} s")
            start = Timer()
            if test_set:
                self.X_train_feat = self.fruit.transform(self.X_train)
            else:
                self.X_train_feat = self.fruit.transform(self.X_train,
                        callbacks=[self._callback])
            print(f"Transforming training set took {Timer() - start} s")
            start = Timer()
            if test_set:
                self.X_test_feat = self.fruit.transform(self.X_test,
                        callbacks=[self._callback])
            else:
                self.X_test_feat = self.fruit.transform(self.X_test)
            print(f"Transforming testing set took {Timer() - start} s")
            self._extracted = watch_branch

            if test_set:
                self._y = self.y_test
            else:
                self._y = self.y_train

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
                        test_cases: list = None):
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

    def print_fruit(self):
        """Prints a summary of the fruits.Fruit object this Fruitalyser
        was initialized with.
        """
        print(self.fruit.summary())

    def plot_prepared_data(self,
                           dim: int = 0) -> tuple:
        """Plots the prepared data of the specified fruits.Fruit object
        with ``fruitalyser.msplot()``. This method can only be used if
        ``self.classify()`` was called before.
        
        :param dim: Which dimension the 2d-plot should show.,
            defaults to 0
        :type dim: int, optional
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s).
        :rtype: tuple
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        msplot(self._callback._prepared_data[:, dim, :], self._y,
               use_axes=ax)
        fig.suptitle("Prepared Data")
        return fig, ax

    def plot_iterated_sums(self,
                           word_indices: list = None) -> tuple:
        """Plots the iterated sums calculated in the specified
        fruits.Fruit object with ``fruitalyser.msplot()``.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param word_indices: Indices of the words that are used for
            plotting. If ``None``, all words are used., defaults to None
        :type word_indices: list, optional
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots(1, len(word_indices))``
            holding the inserted plot(s).
        :rtype: tuple
        """
        if word_indices is None:
            word_indices = list(range(len(self.words)))
        fig, axs = plt.subplots(len(word_indices), 1, sharex=True)
        fig.suptitle("Iterated Sums")
        for i, index in enumerate(word_indices):
            msplot(self._callback._iterated_sums[index][:, 0, :], self._y,
                   use_axes=axs[i])
            axs[i].set_title(str(self.words[index]))
        return fig, axs

    def plot_features(self,
                      sieve_index: int,
                      word_indices: list = None) -> sns.PairGrid:
        """Plots the features of the watched ``fruits.FruitBranch``
        object. The ``seaborn.pairplot`` function is used to create
        a plot based on a ``pandas.DataFrame`` consisting of the
        following data.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index of the FeatureSieve that is used for
            plotting.
        :type sieve_index: int
        :param word_indices: Indices of the words that are used for
            plotting. If this is a list with 5 integers, the pairplot
            (also called scatter matrix) is a grid with 5 rows and 5
            columns. If ``None``, all words are used., defaults to None
        :type word_indices: list, optional
        :returns: Pairplot of the specified features.
        :rtype: seaborn.PairGrid
        """
        if word_indices is None:
            word_indices = list(range(len(self.words)))
        feats = self.get_feature_dataframe(sieve_index,
                                           word_indices)
        pp = sns.pairplot(feats,
                          hue="Class",
                          diag_kind="hist",
                          markers="+",
                          palette=sns.color_palette("tab10",
                                                    len(set(self._y)))
                         )
        pp.fig.suptitle(f"Features from sieve {sieve_index}")
        return pp

    def get_feature_dataframe(self,
                              sieve_index: int,
                              word_indices: list = None) -> np.ndarray:
        """Returns a ``pandas.DataFrame`` object with all features
        matching the following specifications.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index of the FeatureSieve that is used.
        :type sieve_index: int
        :param word_indices: Indices of the words that are used. The
            names of these objects will be the columns names of the
            table. If ``None``, all words are used., defaults to None
        :type word_indices: list, optional
        :rtype: pandas.DataFrame
        """
        if word_indices is None:
            word_indices = list(range(len(self.words)))
        feat_table = np.array([self._callback._sieved_data[:,
                                self.get_feature_index(sieve_index, i)]
                               for i in word_indices])
        feats = pd.DataFrame(feat_table.T,
                             columns=[str(self.words[i])
                                      for i in word_indices])
        feats["Class"] = self._y
        return feats

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
        pca.fit(self._callback._sieved_data)
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
