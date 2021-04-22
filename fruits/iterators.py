import itertools
import re
import numpy as np

class Monomial:
	"""Class Monomial
	
	An instance of this class is a collection of functions, so called 
	letters. It can be called on a two dimensional numpy array and 
	returns the multiplied outputs of all letters called on the input 
	array.
	"""
	def __init__(self):
		self._letters = []

	def append(self, letter):
		if not callable(letter):
			raise TypeError("Cannot append non-callable object")
		self._letters.append(letter)

	def letters(self):
		for letter in self._letters:
			yield letter

	def __call__(self, X:np.ndarray):
		X_mul = np.ones(X.shape[1])
		for letter in self.letters():
			X_mul *= letter(X)
		return X_mul

class SummationIterator:
	"""Class SummationIterator
	
	This (mostly abstractly) used class is a collection of Monomials.
	"""
	def __init__(self, name:str=""):
		self._name = name
		# array of monomials
		self._monomials = []
		self._monomial_index = 0

	@property
	def name(self) -> str:
		return self._name

	@name.setter
	def name(self, name:str):
		self._name = name

	def __str__(self) -> str:
		return self._name

	def __repr__(self) -> str:
		return "SummationIterator('"+str(self)+"')"

	def append(self, mon:Monomial):
		if not isinstance(mon, Monomial):
			raise TypeError("")
		self._monomials.append(mon)

	def monomials(self):
		for mon in self._monomials:
			yield mon

class SimpleWord(SummationIterator):
	"""Class SimpleWord

	A SimpleWord is a concatenation of monomials that have letters 
	(functions) which extract a single dimension of a multidimesional 
	time series.
	"""
	def __init__(self, string):
		super().__init__()
		if not re.fullmatch(r"(\[\d+\])+", string):
			raise ValueError("SimpleWord can only be initilized with a string "+
							 "matching the regular expression "+
							 r"'(\[\d+\])+'")
		monomials = [x[1:] for x in string.split("]")][:-1]
		for monomial in monomials:
			mon = Monomial()
			counts = {int(letter)-1: monomial.count(letter) for letter in 
					  set([x for x in monomial])}
			for i in counts:
				mon.append(lambda X: X[i, :]**counts[i])
			self.append(mon)

		self.name = string

	def __repr__(self) -> str:
		return "SimpleWord("+str(self)+")"

def generate_random_words(number:int,
						  dim:int=1,
						  monomial_length:int=3,
						  n_monomials:int=3) -> list:
	"""Returns randomly initialized instances of the class SimpleWord.
	
	:param number: number of instances created
	:type number: int
	:param dim: maximal dimensionality the letters of any Monomial in 
	any	SimpleWord can extract, defaults to 1
	:type dim: int, optional
	:param monomial_length: maximal number of letters of any Monomial, 
	defaults to 3
	:type monomial_length: int, optional
	:param n_monomials: maximal number of Monomials of any SimpleWord, 
	defaults to 3
	:type n_monomials: int, optional
	:returns: list of SimpleWords
	:rtype: {list}
	"""
	words = []
	av_elements = [str(i+1) for i in range(dim)]
	for i in range(number):
		length = np.random.randint(1,n_monomials+1)
		conc = ""
		for j in range(length):
			clength = np.random.randint(1,monomial_length+1)
			conc += "["+"".join(np.random.choice(av_elements, size=clength))+"]"
		words.append(SimpleWord(conc))
	return words

def generate_words(dim:int=1,
				   monomial_length:int=1,
				   n_monomials:int=1) -> list:
	"""Returns all possible and unique SimpleWords up to the given 
	boundaries.
	
	:param dim: maximal dimensionality the letters of any Monomial in 
	any	SimpleWord can extract, defaults to 1
	:type dim: int, optional
	:param monomial_length: maximal number of letters of any Monomial, 
	defaults to 1
	:type monomial_length: int, optional
	:param n_monomials: maximal number of Monomials of any SimpleWord, 
	defaults to 1
	:type n_monomials: int, optional
	:returns: list of SimpleWords
	:rtype: {list}
	"""
	monomials = []
	for l in range(1, monomial_length+1):
		mons = list(itertools.combinations_with_replacement(
							list(range(1, dim+1)), l))
		for mon in mons:
			monomials.append(list(mon))

	words = []
	for n in range(1, n_monomials+1):
		words_n = list(itertools.product(monomials, repeat=n))
		for word in words_n:
			words.append("".join([str(x).replace(", ","") for x in word]))

	for i in range(len(words)):
		words[i] = SimpleWord(words[i])

	return words