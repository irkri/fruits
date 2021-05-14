from context import fruits
from configurations import CONFIGURATIONS
import os
import logging
import numpy as np
from timeit import default_timer as Timer
from sklearn.linear_model import RidgeClassifierCV

DATA_PATH = "../../data/"
TABLE_HEADER = "{:=^25}{:=^25}{:=^25}{:=^25}{:=^25}".format("Dataset",
					"Feature Calculation Time",
					"Training Time",
					"Testing Time",
					"Test Accuracy")
OUTPUT_FILE = "ucr_configuration_results.txt"

# empty the output file if it exists already
with open(OUTPUT_FILE, "w") as f:
	f.truncate(0)

# create a logger that flushes accuracy results to the given file path
# at the end of the classification of each dataset
logger = logging.Logger("fruits classification results")
fh = logging.FileHandler(OUTPUT_FILE)
logger.addHandler(fh)

datasets = []
for dir_file in sorted(os.listdir(DATA_PATH)):
		if os.path.isdir(DATA_PATH+dir_file):
			datasets.append(dir_file)

results = np.zeros((len(CONFIGURATIONS), len(datasets), 4))

for k, fruit in enumerate(CONFIGURATIONS):

	logger.info(f"Configuration: {fruit.name}\n")
	logger.info(TABLE_HEADER)

	print(f"Starting: Configuration {k+1}")

	for i, dataset in enumerate(datasets):

		train = np.loadtxt(f"{DATA_PATH}/{dataset}/{dataset}_TRAIN.txt")
		test = np.loadtxt(f"{DATA_PATH}/{dataset}/{dataset}_TEST.txt")

		y_train, X_train = train[:, 0].astype(np.int32), train[:, 1:]
		y_test, X_test = test[:, 0].astype(np.int32), test[:, 1:]

		start = Timer()
		X_train_feat = fruit(X_train)
		X_test_feat = fruit(X_test)
		results[k, i, 0] = Timer()-start

		start = Timer()
		classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), 
									   normalize = True)
		classifier.fit(X_train_feat, y_train)
		results[k, i, 1] = Timer()-start

		start = Timer()
		results[k, i, 3] = classifier.score(X_test_feat, y_test)
		results[k, i, 2] = Timer()-start
		logger.info("{: ^25}{: ^25}{: ^25}{: ^25}{: ^25}".format(datasets[i],
			f"{round(results[k, i, 0], 3)}", f"{round(results[k, i, 1], 3)}",
			f"{round(results[k, i, 2], 3)}", f"{round(results[k, i, 3], 3)}"))

	logger.info(len(TABLE_HEADER)*"-")
	logger.info("{: ^25}{: ^25}{: ^25}{: ^25}{: ^25}".format("MEAN",
			f"{round(results[k, :, 0].mean(), 6)}",
			f"{round(results[k, :, 1].mean(), 6)}",
			f"{round(results[k, :, 2].mean(), 6)}",
			f"{round(results[k, :, 3].mean(), 6)}"))
	if k<len(CONFIGURATIONS)-1:
		logger.info("\n"+len(TABLE_HEADER)*"*"+"\n")
	
	print(f"Done.")
