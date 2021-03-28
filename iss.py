import numpy as np

def iterated_sums(Z:np.array, composition:str="[1]") -> np.array:
	return np.cumsum(Z)

def get_increments(X:np.array, padding:int=-1) -> np.array:
	if padding in [0,-1]:
		np.insert(X,padding,0)
	return np.convolve(X,np.array([-1,1]))

def ppv(X:np.array) -> float:
	return np.sum(X>=0)/len(X)