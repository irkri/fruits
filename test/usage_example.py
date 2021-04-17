from context import fruits
import numpy as np

# initialize FeatureExtractor
featex = fruits.FeatureExtractor()

np.random.seed(123)
# initialize 1000 random time series of length 100 with 2 dimensions
Y = np.random.rand(200000).reshape(1000, 2, 100)

# data preparation for FeatureExtractor
featex.set_input_data(Y)

# default preperator is the identity function
print("prepared data; using identity function")
print(featex.prepared_data()[0,0,:10])

# try to use the increments of the time series as another DataPreparateur
featex.add_data_preparateur(fruits.preparateurs.INC)

print("\nprepared data; using increments function")
print(featex.prepared_data()[0,0,:10])

# define SummationIterators to be used for the Iterated Sums
sumit1 = fruits.SummationIterator.build_from_concatenationstring("[112][2212]")
sumit2 = fruits.SummationIterator.build_from_concatenationstring("[12][111]")
iterators = fruits.iterators.generate_random_concatenations(
			number=8, dim=2, max_letter_weight=3, max_concatenation_length=5)
featex.add_summation_iterator(sumit1)
featex.add_summation_iterator(sumit2)
for it in iterators:
	featex.add_summation_iterator(it)

print("\nUsed Concatenations:")
print(sumit1)
print(sumit2)
for it in iterators:
	print(it)

# calculate iterated sums
iss = featex.iterated_sums()
print(f"\niterated sums; output shape with 10 SummationIterators: {iss.shape}")
print(iss[0,0,:10])

# add 3 features for testing purposes
featex.add_feature(fruits.features.PPV)
featex.add_feature(fruits.features.MAX)
featex.add_feature(fruits.features.MIN)

# calculate features
print(f"\nTotal number of features: {10}*{3}={10*3}")
features = featex.features()
print(f"features; output shape using 3 different feature \
functions: {features.shape}")
print(features[:1,:])