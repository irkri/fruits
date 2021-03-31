from context import iss

import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_names", required = True)
parser.add_argument("-i", "--input_path", required = True)
parser.add_argument("-o", "--output_path", required = True)
parser.add_argument("-c", "--num_compositions", type = int, default= 100)

arguments = parser.parse_args()

dataset_names = np.loadtxt(arguments.dataset_names, "str")
results = pd.DataFrame(index = dataset_names,
					   columns = ["accuracy_mean",
								  "accuracy_standard_deviation",
								  "time_training_seconds",
								  "time_test_seconds"],
					   data = 0)
results.index.name = "dataset"

for name in dataset_names:
	train_data = np.loadtxt(
			f"{arguments.input_path}/{name}/{name}_TRAIN.txt")
	Y_train = train_data[:, 0]
	X_train = train_data[:, 1:]

	test_data = np.loadtxt(
			f"{arguments.input_path}/{name}/{name}_TEST.txt")
	Y_test = test_data[:, 0]
	X_test = test_data[:, 1:]

	concatinations = iss.generate_concatinations(
						number=arguments.num_compositions, dim=1, 
						max_concatination_length=10,
						max_composition_length=5)

	features_test = iss.features_from_iterated_sums(X_test, concatinations)
	features_train = iss.features_from_iterated_sums(X_train, concatinations)

	classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), 
								   normalize = True)
	classifier.fit(features_train, Y_train)

	_results = np.array([classifier.score(features_test, Y_test)])

	results.loc[name, "accuracy_mean"] = _results.mean()
	# results.loc[name, "accuracy_standard_deviation"] = _results.std()
	# results.loc[name, "time_training_seconds"] = _timings.mean(1)[[0, 2]].sum()
	# results.loc[name, "time_test_seconds"] = _timings.mean(1)[[1, 3]].sum()

results.to_csv(f"{arguments.output_path}/results_ucr.csv")

