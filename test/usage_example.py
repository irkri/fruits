from context import fruits
import numpy as np

# initialize Fruit object
featex = fruits.Fruit()

np.random.seed(123)
# initialize 1000 random time series of length 100 with 2 dimensions
X = np.random.rand(200000).reshape(1000, 2, 100)

# specify input data
featex.set_input_data(X)

# default preperator is the identity function
print("prepared data; using identity function (i.e. doing nothing), "
	  f"input shape: {X.shape}")
print(featex.prepared_data()[0,0,:10])

# try to use the increments of the time series as another DataPreparateur
featex.add(fruits.preparateurs.INC)

print(f"\nprepared data; using {fruits.preparateurs.INC}; "
	  f"output shape: {featex.prepared_data().shape}")
print(featex.prepared_data()[0,0,:10])

# define SummationIterators to be used for the Iterated Sums
# the simplest construct is a SimpleWord
sumit1 = fruits.iterators.SimpleWord("[112][2212]")
sumit2 = fruits.iterators.SimpleWord("[12][111]")

# ... or let fruits choose random SimpleWords
iterators = fruits.iterators.generate_random_words(
				number=8, dim=2, monomial_length=3, n_monomials=5)

# add function for the Fruit object can be used in many different ways
featex.add(sumit1)
featex.add(sumit2)
featex.add(iterators)

print("\nUsed SummationIterators:")
print(sumit1)
print(sumit2)
for it in iterators:
	print(it)

# calculate iterated sums using the above defined SummationIterators
# e.g. [<[112][2212], ISS(X, 0, N)> for N in range(len(X))]
iss = featex.iterated_sums()
print(f"\niterated sums; output shape: {iss.shape}")
print(iss[0,0,:10])

# add 3 features for testing purposes
# PPV is tuneable, the quantile used to calculate this feature can be specified
featex.add(fruits.features.PPV(quantile=0.75))
featex.add(fruits.features.MAX)
featex.add(fruits.features.MIN)

# use built-in methods for the Fruit object to get information about the 
# processed data
print(f"\nTotal number of features: {featex.nfeatures()}")

# calculate the features
features = featex.features()
print(f"features; output shape: {features.shape}")
print(features[0,:])