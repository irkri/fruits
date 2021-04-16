from context import fruits
import numpy as np

'''
Noticed from the tests so far:
- a padding in the calculation of incremented sums is crucial for the
  resulting features (Inc[0] = X[0] leads in most cases to a ppv of 1)
- there might be a correlation between shorter concatinations and higher ppvs
'''

X = np.random.rand(20)
print(f"X = {X}\n")

Z = fruits.get_increments(X)
print(f"Increments of X are:\nZ = {Z}\n")

conc = "[11][111]"
iterator = fruits.SummationIterator.build_from_compositionstring(conc)

ISS = fruits.iterated_sums(Z, iterator)
print(f"<{conc},ISS(Z)> = {ISS}")
print(f"Proportion of positive values: {fruits.ppv(ISS)}")

print("\n"+40*"-"+"\n")

num_concs = 10
print(f"Generating {num_concs} random concatinations...")
concs = fruits.generate_concatinations(num_concs, dim=1,
									max_concatenation_length=20,
									max_composition_length=3)
print("Done.")

num_input = 100
length_time_series = 100
X_10000 = np.random.rand(num_input*length_time_series)
X_10000 = X_10000.reshape(num_input,length_time_series)
print(f"Calculating increments for all {num_input} time series of length \
{length_time_series}...")
for i in range(num_input):
	X_10000[i] = fruits.get_increments(X_10000[i])
print("Done.")
# print(f"Calculating corresponding features of Iterated Sums...")
# results = fruits.features_from_iterated_sums(X_10000, concs)
# print("Done.")
# sample_size = 3
# print(f"All {num_concs} features of the first {sample_size} time series:\n")
# for i in range(sample_size):
# 	print(results[i])