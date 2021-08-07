"""This python module contains a class FRUITSExperiment, that can be
used to perform a classification task for multidimensional time series
data using a Fruit feature extractor from the package fruits.
A comet_ml experiment can be supplied to the class for tracking the
results of the experiment.
"""

import os
import time
from timeit import default_timer as Timer

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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
        "Shape",
        "FRUITS Time",
        "FRUITS Acc",
        "ROCKET Acc",
        "M",
    ]

    def __init__(self,
                 rocket_csv=None,
                 verbose=True,
                 classifier=None,
                 scaler=None,
                 comet_experiment=None):
        self._results = pd.DataFrame(columns=self.output_header_names)
        if rocket_csv is not None:
            self._rocket_csv = pd.read_csv(rocket_csv)
        else:
            self._rocket_csv = rocket_csv
        self._verbose = verbose
        # dictionary self._datasets[path] = [datasets in path]
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

    def append_data(self, path: str, names: list = None):
        """Finds time series datasets in the specified directory for
        later usage.
        
        :param path: Path of a directory that contains folders
            structured like the datasets you get from
            timeseriesclassification.com.
        :type path: str
        :param names: List of dataset names. Only the datasets in
            this list will be remembered. If `None`, then all datasets
            are used., defaults to None
        :type names: list of strings
        """
        if not path.endswith("/"):
            path += "/"
        self._datasets[path] = []
        for folder in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, folder)):
                if names is None or folder in names:
                    self._datasets[path].append(folder)
        if len(self._datasets[path]) == 0:
            del self._datasets[path]

    def classify(self, fruit: fruits.Fruit):
        """Classifies all datasets added earlier and summarizes the
        results in a pandas DataFrame.
        
        :param fruit: Feature extractor to use for classification.
        :type fruits: fruits.Fruit
        """
        self._fruit = fruit

        if self._comet_exp is not None:
            self._comet_exp.log_dataset_info(name=\
                ",".join([ds for path in self._datasets
                             for ds in self._datasets[path]]))
            self._comet_exp.log_text(fruit.summary())

        if self._verbose:
            print(f"Starting Classification")

        i = 0

        for j, path in enumerate(self._datasets):

            for dataset in self._datasets[path]:

                results = [dataset]

                if self._comet_exp is not None:
                    self._comet_exp.set_step(i)

                X_train, y_train, X_test, y_test = tsdata.load_dataset(
                    path+dataset)
                results.append(f"{X_train.shape[0]}/{X_test.shape[0]}/"+
                               f"{X_train.shape[2]}")

                start = Timer()
                fruit.fit(X_train)
                X_train_feat = fruit.transform(X_train)
                X_test_feat = fruit.transform(X_test)
                results.append(Timer() - start)

                self._scaler.fit(X_train_feat)
                X_train_feat_scaled = self._scaler.transform(
                                                X_train_feat)
                X_test_feat_scaled = self._scaler.transform(
                                                X_test_feat)

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
                if results[3] >= results[4]:
                    mark = "X"
                results.append(mark)
                self._results.loc[len(self._results)] = results
                if self._verbose:
                    print(".", end="", flush=True)
                i += 1

        if self._comet_exp is not None:
            self._comet_exp.log_table("results.csv", self._results)
            self._comet_exp.log_text(self._results.to_markdown(index=False,
                                                        numalign="center",
                                                        stralign="center"))
            self._comet_exp.log_html(self._results.to_html(index=False,
                                                           justify="center"))

        if self._verbose:
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
        if self._fruit is None:
            raise RuntimeError("No output available")
        filename = filename.split(".")[0]

        if os.path.isfile(filename+".txt") or os.path.isfile(filename+".csv"):
            filename += "-" + time.strftime("%Y-%m-%d-%H%M%S")

        if txt:
            with open(filename+".txt", "w") as file:
                file.write(self._results.to_markdown(index=False,
                                                     tablefmt="grid",
                                                     numalign="center",
                                                     stralign="center"))
                file.write("\n\n\n"+self._fruit.summary()+"\n")
        if csv:
            self._results.to_csv(filename+".csv", index=False)
