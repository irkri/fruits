"""This python module contains a class FRUITSExperiment, that can be
used to perform a classification task for multidimensional time series
data using a Fruit feature extractor from the package fruits.
A comet_ml experiment can be supplied to the class for tracking the
results of the experiment.
"""

import os
import time
from typing import Union, List
from timeit import default_timer as Timer

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import FunctionTransformer

from context import fruits
import tsdata

class FRUITSExperiment:
    """A pipeline that connects feature extraction of the fruits package
    with a classifier from sklearn for a classification task of 
    multidimensional time series data.
    
    :param rocket_csv: String of the path to a csv file with
        the classification results from ROCKET. These results will be
        used for comparision. If None is given, a 0 will be inserted at
        each point in the output table., defaults to None
    :type rocket_csv: str, optional
    :param verbose: If `True`, print status of classfication to the
        console., defaults to `True`
    :type verbose: bool, optional
    :param classifier: Classifier with fit and score method from
        sklearn., defaults to
        ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                            normalize=True)``
    :type classfier: ClassifierMixin from sklearn, optional
    :param scaler: An object that allows calls of fit() and transform()
        and that scales the features extracted by the fruits pipeline.,
        defaults to an identity scaler (that doesn't transform at all)
    :type scaler: Some scaler, preferably from sklearn., optional
    :param comet_experiment: Experiment object from the package comet_ml
        that is used for saving the results from the classification.,
        defaults to None
    :type comet_experiment: comet_ml.Experiment, optional
    """
    output_header_names = [
        "Dataset",
        "TrS",
        "TeS",
        "Dim",
        "Len",
        "FRUITS Time",
        "FRUITS Acc",
        "ROCKET Acc",
        "M",
    ]

    def __init__(self,
                 rocket_csv=None,
                 classifier=None,
                 scaler=None,
                 comet_experiment=None):
        self._results = pd.DataFrame(columns=self.output_header_names)
        if rocket_csv is not None:
            self._rocket_csv = pd.read_csv(rocket_csv)
        else:
            self._rocket_csv = rocket_csv
        # dictionary self._datasets[path] = ([datasets in path], univariate?)
        self._datasets = dict()
        self._comet_exp = comet_experiment
        self._fruit = None
        if classifier is None:
            self._classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                                 normalize=True)
        else:
            self._classifier = classifier
        if scaler is None:
            self._scaler = FunctionTransformer(lambda X: X, validate=False)
        else:
            self._scaler = scaler

    def append_data(self,
                    path: str,
                    names: Union[List[str], str, None] = None,
                    univariate: bool = True):
        """Finds time series datasets in the specified directory for
        later usage.
        
        :param path: Path of a directory that contains folders
            structured like the datasets you get from
            timeseriesclassification.com.
        :type path: str
        :param names: List of dataset names. Only the datasets in
            this list will be remembered. If ``None``, then all datasets
            are used. If ``names`` is a single string, it will be
            treated as a text-file name where all dataset names are
            listed and seperated by a newline character.,
            defaults to None
        :type names: Union[List[str], str]
        :param univariate: If ``True``, the data in ``path`` is assumed
            to be univariate. Set this option to False for multivariate
            data., defaults to True
        :type univariate: bool, optional
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

    def classify(self,
                 fruit: Union[fruits.Fruit, None] = None,
                 fit_sample_size: Union[float, int] = 1,
                 cache_results: Union[str, None] = None,
                 verbose: bool = True):
        """Classifies all datasets added earlier and summarizes the
        results in a pandas DataFrame.
        
        :param fruit: Feature extractor to use for classification.
            If set to ``None``, the class uses ``fruits.build`` for
            building a fruit for every dataset., defaults to None
        :type fruit: Union[fruits.Fruit, None], optional
        :param fit_sample_size: Option ``sample_size`` supplied to the
            fitting method of the fruit., defaults to 1
        :type fit_sample_size: Union[float, int], optional
        :param cache_results: If set to a filename, the method will save
            results at the end of each dataset classification to a csv
            file with that name., defaults to None
        :type cache_results: Union[str, None], defaults to None
        """
        if cache_results is not None and cache_results.endswith(".csv"):
            cache_results = cache_results[:-4]
        self._results = pd.DataFrame(columns=self.output_header_names)
        self._fruit = fruit

        if self._comet_exp is not None:
            self._comet_exp.log_dataset_info(name=\
                ",".join([ds for path in self._datasets
                             for ds in self._datasets[path][0]]))
            if self._fruit is not None:
                self._comet_exp.log_text(fruit.summary())

        if verbose:
            print(f"Starting Classification")

        i = 0

        for j, path in enumerate(self._datasets):

            univariate = self._datasets[path][1]

            for dataset in self._datasets[path][0]:

                results = [dataset]

                if self._comet_exp is not None:
                    self._comet_exp.set_step(i)

                X_train, y_train, X_test, y_test = tsdata.load_dataset(
                    path+dataset, univariate=univariate)
                X_train = tsdata.nan_to_num(X_train)
                X_test = tsdata.nan_to_num(X_test)
                results.append(X_train.shape[0])
                results.append(X_test.shape[0])
                results.append(X_train.shape[1])
                results.append(X_train.shape[2])

                if self._fruit is None:
                    fruit = fruits.build(X_train)

                start = Timer()
                fruit.fit(X_train, fit_sample_size)
                X_train_feat = fruit.transform(X_train)
                X_test_feat = fruit.transform(X_test)
                results.append(Timer() - start)

                self._scaler.fit(X_train_feat)
                X_train_feat_scaled = np.nan_to_num(self._scaler.transform(
                                                        X_train_feat))
                X_test_feat_scaled = np.nan_to_num(self._scaler.transform(
                                                        X_test_feat))

                self._classifier.fit(X_train_feat_scaled, y_train)

                results.append(self._classifier.score(
                                X_test_feat_scaled, y_test))

                mark = "-"
                if self._rocket_csv is not None:
                    results.append(self._rocket_csv[
                                    self._rocket_csv["dataset"] == dataset]\
                                    ["accuracy_mean"].iloc[0])
                else:
                    results.append(0)
                if results[6] >= results[7]:
                    mark = "X"
                results.append(mark)
                self._results.loc[len(self._results)] = results
                if verbose:
                    print(".", end="", flush=True)
                if cache_results is not None:
                    self._results.to_csv(f"{cache_results}.csv")
                    if self._comet_exp is not None:
                        self._comet_exp.log_table("results.csv", self._results)
                i += 1

        if self._comet_exp is not None:
            self._comet_exp.log_table("results.csv", self._results)
            self._comet_exp.log_text(self._results.to_markdown(index=False,
                                                        numalign="center",
                                                        stralign="center"))
            self._comet_exp.log_html(self._results.to_html(index=False,
                                                           justify="center"))

        if verbose:
            print(f"\nDone")

    def produce_output(self,
                       filename: str,
                       txt: bool = True,
                       csv: bool = False):
        """Outputs all results of classified datasets to the file(s).
        
        :param filename: Name of the file (without extension) that is
            used for the different extension types. If it exists already
            as a .txt or .csv file, a timestamp is appended to the name
            before saving.
        :type filename: str
        :param txt: If True, a .txt file will be produced containing a
            table of accuracy results and a summary of the used fruits
            object., defaults to True
        :type txt: bool, optional
        :param csv: If True, a .csv file will be produced containing all
            accuracy results., defaults to False
        :type csv: bool, optional
        """
        filename = filename.split(".")[0]

        if os.path.isfile(filename+".txt") or os.path.isfile(filename+".csv"):
            filename += "-" + time.strftime("%Y-%m-%d-%H%M%S")

        if txt:
            with open(filename+".txt", "w") as file:
                file.write(self._results.to_markdown(index=False,
                                                     tablefmt="grid",
                                                     numalign="center",
                                                     stralign="center"))
                file.write("\n\nAverage FRUITS Accuracy: "+
                           str(self._results[
                               self.output_header_names[6]].to_numpy().mean()))
                file.write("\nAverage ROCKET Accuracy: "+
                           str(self._results[
                               self.output_header_names[7]].to_numpy().mean()))
                if self._fruit is not None:
                    file.write("\n\n\n"+self._fruit.summary()+"\n")
        if csv:
            self._results.to_csv(filename+".csv", index=False)
