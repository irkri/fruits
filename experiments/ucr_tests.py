"""Classification of UCR datasets.

Usage:
	$ python ucr_tests.py -d [path to datasets]

Optional arguments:
	-u [text file of datasets if not all datasets in specified folder
	should be used], default = every dataset in directory is used
	-o [directory to put the output file in], default = "./"
	-n [number of runs per dataset], default = 10
"""

from context import fruits
import argparse
from timeit import default_timer as Timer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
import re
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasets_path", required=True)
parser.add_argument("-u", "--use_datasets", default="")
parser.add_argument("-o", "--output_path", default="")
parser.add_argument("-n", "--num_runs", type=int, default=10)

arguments = parser.parse_args()

datasets = []
if arguments.use_datasets=="":
	for dir_file in sorted(os.listdir(arguments.datasets_path)):
		if os.path.isdir(arguments.datasets_path+dir_file):
			datasets.append(dir_file)
else:
	with open(arguments.use_datasets, "r") as f:
		datasets = re.split(r"\s+", f.read())

# configure Fruit object

featex = fruits.Fruit()

featex.add(fruits.preparateurs.STD)
featex.add(fruits.preparateurs.INC)

featex.add(fruits.iterators.generate_words(1, 4, 4))

DIFF = fruits.features.FeatureSieve("slope")
DIFF.set_function(lambda X: X[:, -1]-X[:, 0])

featex.add(fruits.features.PPV(quantile=0.2, constant=False, sample_size=0.2))
featex.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=0.2))
featex.add(fruits.features.PPV(quantile=0.8, constant=False, sample_size=0.2))
featex.add(fruits.features.MAX)
featex.add(fruits.features.MIN)
featex.add(DIFF)

results = np.zeros((len(datasets), arguments.num_runs, 4))

# perform runs for each dataset

for i, dataset in enumerate(datasets):
	print("{:=^80}".format(f"Dataset {dataset}"))

	train = np.loadtxt(f"{arguments.datasets_path}/{dataset}/"+
					   f"{dataset}_TRAIN.txt")
	test = np.loadtxt(f"{arguments.datasets_path}/{dataset}/"+
					  f"{dataset}_TEST.txt")

	y_train, X_train = train[:, 0].astype(np.int32), train[:, 1:]
	y_test, X_test = test[:, 0].astype(np.int32), test[:, 1:]

	for j in range(arguments.num_runs):

		start = Timer()
		train_features = featex(X_train)
		test_features = featex(X_test)
		results[i, j, 0] = Timer()-start

		start = Timer()
		# choose a classifier
		classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
		classifier.fit(train_features, y_train)
		results[i, j, 1] = Timer()-start

		start = Timer()
		results[i, j, 3] = classifier.score(test_features, y_test)
		results[i, j, 2] = Timer()-start
	
	print("{:=^80}\n".format("Done."))

# output formatting

out = "Time series classification using fruits"
out += "\n\n"+80*"-"+"\n\n"
out += "Preparateurs:\n"
for prep in featex.get_data_preparateurs():
	out += f"\t+ {prep}\n"
out += "Iterators:\n"
for smit in featex.get_summation_iterators():
	out += f"\t+ {smit}\n"
out += "Filters:\n"
for fltr in featex.get_feature_sieves():
	out += f"\t+ {fltr}\n"

out += "\n"+80*"-"+"\n\n"

header = "{:=^25}{:=^25}{:=^25}{:=^25}{:=^25}".format("Dataset", 
												"Feature Calculation Time",
												"Training Time", 
												"Testing Time", 
												"Test Accuracy")

out += header+"\n"

for i in range(len(datasets)):
	out += "{: ^25}{: ^25}{: ^25}{: ^25}{: ^25}\n".format(datasets[i], 
f"{round(results[i, :, 0].mean(), 3)} +- {round(results[i, :, 0].std(), 3)}", 
f"{round(results[i, :, 1].mean(), 3)} +- {round(results[i, :, 1].std(), 3)}", 
f"{round(results[i, :, 2].mean(), 3)} +- {round(results[i, :, 2].std(), 3)}", 
f"{round(results[i, :, 3].mean(), 3)} +- {round(results[i, :, 3].std(), 3)}")

out += len(header)*"-"+"\n"

out += "{: ^25}{: ^25}{: ^25}{: ^25}{: ^25}".format("MEAN", 
f"{round(results[:, :, 0].mean(axis=1).mean(axis=0), 6)}", 
f"{round(results[:, :, 1].mean(axis=1).mean(axis=0), 6)}", 
f"{round(results[:, :, 2].mean(axis=1).mean(axis=0), 6)}", 
f"{round(results[:, :, 3].mean(axis=1).mean(axis=0), 6)}")

print("\nAll datasets have been classified. A summary is written to the file "+
	  arguments.output_path+"fruits_results.txt")

with open(arguments.output_path+"fruits_results.txt", "w") as f:
	f.write(out)