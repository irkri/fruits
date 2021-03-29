import numpy as np
from timeit import default_timer as timer

def iterated_sums(Z:np.array, 
				  composition:str="[1]", 
				  verbose:bool=False) -> np.array:
	''' calculates the iterated sums signature for a given input series Z,
	e.g. <[1][12][22],ISS(Z)> = CS(CS(CS(Z[0])*Z[0]*Z[1])*Z[2]^2)
	(CS is np.cumsum) '''
	if verbose:
		print("{:=^80}".format("Iterated Sums Signature"))
		print(f"Input: <{composition},ISS(Z)>\nwith Z={Z}\n")
		_start = timer()
	iterations = [x[1:] for x in composition.split("]")][:-1]
	assert iterations, "Invalid composition"
	if len(Z.shape)==1:
		Z = Z.reshape(len(Z),1)
	P = np.ones(Z.shape[0])
	for i in range(len(iterations)):
		C = np.ones(Z.shape[0])
		for letter in iterations[i]:
			C *= Z[:,int(letter)-1]
		P = np.cumsum(P*C)
		if verbose:
			print("{:-^40}".format("Iteration "+str(i+1)))
			print(f"Composition: [{iterations[i]}]")
			print(f"P: {P}")
			print(f"C: {C}")
	if verbose:
		print("\nDone.")
		print("Time needed: {:.5f}s".format(timer()-_start))
		print("{:=^80}".format(""))
	return P

def get_increments(X:np.array, padding:str="none") -> np.array:
	if padding=="left":
		X = np.insert(X,0,0)
	elif padding=="right":
		X = np.insert(X,len(X),0)
	return np.convolve(X,np.array([1,-1]),mode="valid")

def ppv(X:np.array) -> float:
	if len(X)==0:
		return 0
	return np.sum(X>=0)/len(X)