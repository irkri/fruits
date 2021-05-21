import os
import logging
from timeit import default_timer as Timer

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from context import fruits
from configurations import CONFIGURATIONS

DATA_PATH = "../../data/"
OUTPUT_FILE = "ucr_configuration_results.txt"
ROCKET_RESULTS_FILE = "rocket_results_ucr.csv"
TABLE_HEADER = "{:=^25}{:=^25}{:=^25}{:=^25}".format(
                    "Dataset",
                    "Feature Calculation Time",
                    "Test Accuracy",
                    "Rocket Mean Accuracy")

# empty the output file if it exists already
with open(OUTPUT_FILE, "w") as f:
    f.truncate(0)

rocket_results = pd.read_csv(ROCKET_RESULTS_FILE)

# create a logger that flushes accuracy results to the given file path
# at the end of the classification of each dataset
logger = logging.Logger("fruits classification results")
fh = logging.FileHandler(OUTPUT_FILE)
logger.addHandler(fh)

datasets = []
for dir_file in sorted(os.listdir(DATA_PATH)):
        if os.path.isdir(DATA_PATH + dir_file):
            datasets.append(dir_file)

results = np.zeros((len(CONFIGURATIONS), len(datasets), 4))

for k, fruit in enumerate(CONFIGURATIONS):

    logger.info(f"Configuration: {fruit.name}, " +
                f"Features: {fruit.nfeatures()}\n")
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
        results[k, i, 0] = Timer() - start

        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), 
                                       normalize = True)
        classifier.fit(X_train_feat, y_train)

        results[k, i, 1] = classifier.score(X_test_feat, y_test)
        logger.info("{: ^25}{: ^25}{: ^25}{: ^25}".format(
            dataset,
            round(results[k, i, 0], 3),
            round(results[k, i, 1], 3),
            round(rocket_results[rocket_results["dataset"] == \
                dataset]["accuracy_mean"].to_numpy()[0], 3)))
        logger.handlers[0].flush()

    logger.info(len(TABLE_HEADER) * "-")
    logger.info("{: ^25}{: ^25}{: ^25}{: ^25}".format(
            "MEAN",
            round(results[k, :, 0].mean(), 6),
            round(results[k, :, 1].mean(), 6),
            round(rocket_results["accuracy_mean"].to_numpy().mean(), 6)))

    if k < len(CONFIGURATIONS)-1:
        logger.info("\n" + len(TABLE_HEADER)*"*" + "\n")
    
    print(f"Done.")
