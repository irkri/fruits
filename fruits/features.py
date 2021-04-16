import numpy as np

def feature(func):
	''' decorator function for a single feature extractor;
	each function defined with this decorator takes in a numpy array of one 
	dimension and returns a single number '''
	def wrapper(*args, **kwargs):
		result = func(*args, **kwargs)
		return result
	return wrapper

@feature
def ppv(X:np.ndarray) -> float:
	if len(X)==0:
		return 0
	return np.sum(X>=0)/len(X)