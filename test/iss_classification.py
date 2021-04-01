from context import iss

import argparse
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_names", required=True)
parser.add_argument("-i", "--in_path", required=True)
parser.add_argument("-o", "--out_path", required=True)
parser.add_argument("-c", "--num_compositions", type=int, default=100)
parser.add_argument("-n", "--num_runs", type=int, default=50)

arguments = parser.parse_args()

''' 
start this file by using the above parsed arguments, e.g.
	$ python iss_classification.py -d datasets_used.txt -i ../../ -o .
'''

dataset_names = np.loadtxt(arguments.dataset_names, "str")
results = pd.DataFrame(index = dataset_names,
					   columns = ["accuracy_mean",
								  "accuracy_standard_deviation",
								  "time_training_seconds",
								  "time_test_seconds"],
					   data = 0)
results.index.name = "dataset"

for name in dataset_names:

	_timings = np.zeros((4, arguments.num_runs))
	_results = np.zeros(arguments.num_runs)

	for i in range(arguments.num_runs):

		train_data = np.loadtxt(f"{arguments.in_path}/{name}/{name}_TRAIN.txt")
		Y_train = train_data[:, 0]
		X_train = train_data[:, 1:]

		test_data = np.loadtxt(f"{arguments.in_path}/{name}/{name}_TEST.txt")
		Y_test = test_data[:, 0]
		X_test = test_data[:, 1:]

		concatinations = iss.generate_concatinations(
							number=arguments.num_compositions, dim=1, 
							max_concatination_length=10,
							max_composition_length=4)

		X_test = iss.get_increments(X_test, axis=(1))
		X_train = iss.get_increments(X_train, axis=(1))

		start = timer()
		features_test = iss.features_from_iterated_sums(X_test, concatinations)
		_timings[0, i] = timer()-start

		start = timer()
		features_train = iss.features_from_iterated_sums(X_train, 
														 concatinations)
		_timings[1, i] = timer()-start

		classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), 
									   normalize = True)

		start = timer()
		classifier.fit(features_train, Y_train)
		_timings[2, i] = timer()-start

		start = timer()
		_results[i] = classifier.score(features_test, Y_test)
		_timings[3, i] = timer()-start

	results.loc[name, "accuracy_mean"] = _results.mean()
	results.loc[name, "accuracy_standard_deviation"] = _results.std()
	results.loc[name, "time_training_seconds"] = _timings.mean(1)[[0, 2]].sum()
	results.loc[name, "time_test_seconds"] = _timings.mean(1)[[1, 3]].sum()

results.to_csv(f"{arguments.out_path}/results_ucr.csv")