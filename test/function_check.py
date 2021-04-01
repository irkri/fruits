from context import iss
import numpy as np

'''
Noticed from the tests so far:
- a padding in the calculation of incremented sums is crucial for the
  resulting features (Inc[0] = X[0] leads in most cases to a ppv of 1)
- there might be a correlation between shorter concatinations and higher ppvs
'''

X = np.random.rand(20)
print(f"X = {X}\n")

Z = iss.get_increments(X)
print(f"Increments of X are:\nsZ = {Z}\n")

comp = iss.Composition("[11]")
conc = iss.Concatination.from_str("[11][1]")
conc *= comp

ISS = iss.iterated_sums(Z, conc)
print(f"<{conc},ISS(Z)> = {ISS}")
print(f"Proportion of positive values: {iss.ppv(ISS)}")

print("\n"+40*"-"+"\n")

num_concs = 10
print(f"Generating {num_concs} random concatinations...")
concs = iss.generate_concatinations(num_concs, dim=1,
									max_concatination_length=20,
									max_composition_length=3)
print("Done.")
print(f"First random concatination: {[str(c) for  c in concs]}")

num_input = 100
length_time_series = 100
X_10000 = np.random.rand(num_input*length_time_series).reshape(num_input,
															length_time_series)
print(f"Calculating increments for all {num_input} time series of length \
{length_time_series}...")
for i in range(num_input):
	X_10000[i] = np.insert(iss.get_increments(X_10000[i]),0,0)#X_10000[i][0])
print("Done.")
print(f"Calculating corresponding features of Iterated Sums...")
results = iss.features_from_iterated_sums(X_10000, concs)
print("Done.")
print(f"All {num_concs} features of the first 3 time series:\n")
for i in range(3):
	print(results[i])