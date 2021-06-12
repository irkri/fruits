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
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV

from context import fruits

def msplot(X: np.ndarray,
           y: np.ndarray,
           above_error: bool = True,
           below_error: bool = True,
           figure_height: int = 5,
           figure_width: int = 10) -> tuple:
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
    :returns: Tuple (figure, axis) corresponding to the return value of
        ``matplotlib.pyplot.subplots(
            figsize=(figure_width, figure_height))`` with the newly
            inserted plot(s).
    :rtype: tuple
    """
    classes = sorted(list(set(y)))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
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
    return (fig, ax)

def load_dataset(path: str) -> tuple:
    """Returns a time series dataset that is formatted as a .txt file
    and readable with numpy.
    
    :param path: Path to the dataset. The path has to point to a
        folder 'name' with two files:
        'name_TEST.txt' and 'name_TRAIN.txt' where 'name' is the name of
        the dataset.
    :type path: str
    :returns: Tuple (X_train, y_train, X_test, y_test)
    :rtype: tuple
    """
    dataset = path.split("/")[-1]
    train_raw = np.loadtxt(f"{path}/{dataset}_TRAIN.txt")
    test_raw = np.loadtxt(f"{path}/{dataset}_TEST.txt")
    X_train = train_raw[:, 1:]
    y_train = train_raw[:, 0].astype(np.int32)
    X_test = test_raw[:, 1:]
    y_test =  test_raw[:, 0].astype(np.int32)
    return X_train, y_train, X_test, y_test

class Fruitalyser:
    """Class for analysing results for a fruits.Fruit object and its
    classification pipeline.

    :param fruits_configuration: A fruits configuration for feature
        extraction
    :type fruits_configuration: fruits.Fruit
    :param data: Tuple of a (already split) classification dataset
        ``[X_train, y_train, X_test, y_test]``. This format is the same
        you get from calling the module method
        ``fruitalyser.load_dataset``.
    :type data: tuple
    """
    def __init__(self,
                 fruits_configuration: fruits.Fruit,
                 data: tuple):
        self.fruit = fruits_configuration
        self._extracted = None
        self.X_train , self.y_train, self.X_test, self.y_test = data

    def classify(self,
                 classifier=None,
                 watch_branch: int = 0):
        """Classifies the specified data by first extracting the
        features of the time series using the fruits.Fruit object.
        
        :param classifier: Used classifier, defaults to
            ```RidgeClassifierCV(alphas=np.logspace(-3,3,10), 
                                 normalize=True)```
        :type classifier: Classifier from the package sklearn with
            a fit and score method., optional
        :param watch_branch: The incremental steps (prepared data, 
            iterated sums) of a fruits.Fruit object can be only saved
            for one branch in this object. The index of this branch
            is specified with this argument.
            All future calls of other methods on this Fruitalyser
            object will depend on this option., defaults to 0
        :type watch_branch: int, optional
        """
        if self._extracted is None or self._extracted != watch_branch:
            watched_branch = self.fruit.branches()[watch_branch]
            start = Timer()
            self.fruit.fit(self.X_train)
            print(f"Fitting took {Timer() - start} s")
            start = Timer()
            self.all_X_train_feat = self.fruit.transform(self.X_train)
            print(f"Transforming training set took {Timer() - start} s")
            self.X_train_prep = watched_branch._prepared_data.copy()
            self.X_train_iter = watched_branch._iterated_data.copy()
            self.X_train_feat = watched_branch._sieve(
                                    watched_branch._iterated_data)
            start = Timer()
            self.all_X_test_feat = self.fruit.transform(self.X_test)
            print(f"Transforming testing set took {Timer() - start} s")
            self.X_test_prep = watched_branch._prepared_data.copy()
            self.X_test_iter = watched_branch._iterated_data.copy()
            self.X_test_feat = watched_branch._sieve(
                                    watched_branch._iterated_data)

            self.fruit.clear_cache()
            self._extracted = watch_branch

            self.preparateurs = watched_branch.get_preparateurs()
            self.iterators = watched_branch.get_iterators()
            self.sieves = watched_branch.get_sieves()
            self.nfeatures = sum([sieve.nfeatures() for sieve in self.sieves])
        else:
            print("Features already extracted.")

        if classifier is None:
            classifier = RidgeClassifierCV(alphas=np.logspace(-3,3,10),
                                           normalize=True)
        classifier.fit(self.all_X_train_feat, self.y_train)
        self.test_score = classifier.score(self.all_X_test_feat,
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
            clssfr.fit(self.all_X_train_feat, self.y_train)
            accuracies[i] = clssfr.score(self.all_X_test_feat, self.y_test)
        fig, ax = plt.subplots()
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

    def plot_input_data(self, test_set: bool = True) -> tuple:
        """Plots the input data with ``fruitalyser.msplot()``.
        
        :param test_set: If True, the test set is used for plotting.,
            defaults to True
        :type test_set: bool, optional
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s).
        :rtype: tuple
        """
        if test_set:
            return msplot(self.X_test, self.y_test)
        return msplot(self.X_train, self.y_train)

    def plot_prepared_data(self, test_set: bool = True) -> tuple:
        """Plots the prepared data of the specified fruits.Fruit object
        with ``fruitalyser.msplot()``. This method can only be used if
        ``self.classify()`` was called before.
        
        :param test_set: If True, the transformed results of the test
            set are used for plotting., defaults to True
        :type test_set: bool, optional
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s).
        :rtype: tuple
        """
        if test_set:
            return msplot(self.X_test_prep[:, 0, :], self.y_test)
        return msplot(self.X_train_prep[:, 0, :], self.y_train)

    def get_prepared_data(self, test_set: bool = True) -> np.ndarray:
        """Returns the prepared data of the fruits.Fruit object.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param test_set: If True, the transformed results of the test
            set are returned., defaults to True
        :type test_set: bool, optional
        :returns: Prepared data of the watched FruitBranch
        :rtype: np.ndarray
        """
        if test_set:
            return self.X_test_prep
        return self.X_train_prep

    def plot_iterated_data(self,
                           iterator_indices: list = None,
                           test_set: bool = True) -> tuple:
        """Plots the iterated sums calculated in the specified
        fruits.Fruit object with ``fruitalyser.msplot()``.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param iterator_indices: Indices of the SummationIterator
            objects that are used for plotting. If ``None``, all
            iterators are used., defaults to None
        :type iterator_indices: list, optional
        :param test_set: If True, the transformed results of the test
            set are used for plotting., defaults to True
        :type test_set: bool, optional
        :returns: Tuple (figure, axis) corresponding to the return value
            of ``matplotlib.pyplot.subplots()`` holding the inserted
            plot(s).
        :rtype: tuple
        """
        if iterator_indices is None:
            iterator_indices = list(range(len(self.iterators)))
        plots = []
        for index in iterator_indices:
            if test_set:
                fig, ax = msplot(self.X_test_iter[:, index, :], self.y_test)
            else:
                fig, ax = msplot(self.X_train_iter[:, index, :], self.y_train)
            ax.set_title(str(self.iterators[index]))
            plots.append((fig, ax))
        return plots

    def get_iterated_data(self,
                          iterator_indices: list = None,
                          test_set: bool = True) -> np.ndarray:
        """Returns the iterated sums calculated in the fruits.Fruit
        object.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param iterator_indices: Indices of the SummationIterator
            objects that are used. If ``None``, all iterators are used.,
            defaults to None
        :type iterator_indices: list, optional
        :param test_set: If True, the transformed results of the test
            set are returned., defaults to True
        :type test_set: bool, optional
        :returns: Iterated sums used in the watched FruitBranch
        :rtype: np.ndarray
        """
        if iterator_indices is None:
            iterator_indices = list(range(len(self.iterators)))
        if test_set:
            return self.X_test_iter[:, iterator_indices, :]
        return self.X_test_iter[:, iterator_indices, :]

    def plot_features(self,
                      sieve_index: int,
                      iterator_indices: list = None,
                      test_set: bool = True) -> sns.PairGrid:
        """Plots the features of the watched ``fruits.FruitBranch``
        object. The ``seaborn.pairplot`` function is used to create
        a plot based on a ``pandas.DataFrame`` consisting of the
        following data.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index of the FeatureSieve that is used for
            plotting.
        :type sieve_index: int
        :param iterator_indices: Indices of the SummationIterator
            objects that are used for plotting. If this is a list with
            5 integers, the pairplot (also called scatter matrix) is
            a grid with 5 rows and 5 columns. If ``None``, all
            iterators are used., defaults to None
        :type iterator_indices: list, optional
        :param test_set: If True, the results from the test set are
            used., defaults to True
        :type test_set: bool, optional
        :returns: Pairplot of the specified features.
        :rtype: seaborn.PairGrid
        """
        if iterator_indices is None:
            iterator_indices = list(range(len(self.iterators)))
        feats = self.get_feature_dataframe(sieve_index,
                                           iterator_indices,
                                           test_set)
        if test_set:
            pp = sns.pairplot(feats,
                              hue="Class",
                              diag_kind="hist",
                              markers="+",
                              palette=sns.color_palette("tab10",
                                        len(set(self.y_test))))
        else:
            pp = sns.pairplot(feats,
                              hue="Class",
                              diag_kind="hist",
                              markers="+",
                              palette=sns.color_palette("tab10",
                                        len(set(self.y_train))))
        return pp

    def get_feature_dataframe(self,
                              sieve_index: int,
                              iterator_indices: list = None,
                              test_set: bool = True) -> np.ndarray:
        """Returns a ``pandas.DataFrame`` object with all features
        matching the following specifications.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param sieve_index: Index of the FeatureSieve that is used.
        :type sieve_index: int
        :param iterator_indices: Indices of the SummationIterator
            objects that are used. The names of these objects will be
            the columns names of the table. If ``None``, all
            iterators are used., defaults to None
        :type iterator_indices: list, optional
        :param test_set: If True, the results from the test set are
            used., defaults to True
        :type test_set: bool, optional
        :rtype: pandas.DataFrame
        """
        if iterator_indices is None:
            iterator_indices = list(range(len(self.iterators)))
        if test_set:
            feat_table = np.array([self.X_test_feat[:,
                                    self.get_feature_index(sieve_index, i)]
                                   for i in iterator_indices])
            feats = pd.DataFrame(feat_table.T,
                                 columns=[str(self.iterators[i])
                                          for i in iterator_indices])
            feats["Class"] = self.y_test
        else:
            feat_table = np.array([self.X_train_feat[:,
                                    self.get_feature_index(sieve_index, i)]
                                   for i in iterator_indices])
            feats = pd.DataFrame(feat_table.T,
                                 columns=[str(self.iterators[i])
                                          for i in iterator_indices])
            feats["Class"] = self.y_train
        return feats

    def get_features(self, test_set: bool = True) -> np.ndarray:
        """Returns the features calculated with the fruits.Fruit object.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param test_set: If True, the transformed results of the test
            set are returned., defaults to True
        :type test_set: bool, optional
        :returns: Features calculated in the watched FruitBranch
        :rtype: np.ndarray
        """
        if test_set:
            return self.X_test_feat
        return self.X_train_feat

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
        pca.fit(self.X_train_feat)
        feature_pc_correlation = pd.DataFrame(pca.components_,
            columns=['Feat-'+str(i+1) for i in range(self.fruit.nfeatures())],
            index=['PC-'+str(i+1) for i in range(components)])
        return feature_pc_correlation

    def rank_iterators_and_sieves(self,
                                  n: int) -> list:
        """Ranks the SummationIterator and FeatureSieve objects by
        relevance for the classification in the watched FruitBranch.
        
        :param n: Number of objects to rank
        :type n: int
        :returns: List of index tuples (SummationIterator, FeatureSieve)
            sorted by relevance for the classfication.
        :rtype: list
        """
        correlation = self.pca_correlation(n)
        sorted_feature_indices = [np.argmax(correlation.iloc[i])
                                  for i in range(n)]
        return [self.split_feature_index(feat)
                for feat in sorted_feature_indices]

    def split_feature_index(self, index: int) -> tuple:
        """For a given index for one of the calculated features from
        the ``fruits.Fruit`` object, this method returns a tuple
        containing the index of the SummationIterator and the index of
        the FeatureSieve that led to this particular feature.
        This method can only be used if ``self.classify()`` was called
        before.
        
        :param index: Feature index
        :type index: int
        :rtype: tuple
        """
        iterator_index = index // self.nfeatures
        sieve_index =  index - self.nfeatures*iterator_index
        return (iterator_index, sieve_index)

    def get_feature_index(self, sieve_index: int, iterator_index: int) -> int:
        """Returns a feature index in the array of all concatenated
        features.
        This method can only be used if ``self.classify()`` was called
        before.

        :param sieve_index: Index of the corresponding FeatureSieve.
        :type sieve_index: int
        :param iterator_index: Index of the corresponding
            SummationIterator.
        :type iterator_index: int
        :rtype: int
        """
        index = iterator_index * self.nfeatures
        return index + sieve_index
