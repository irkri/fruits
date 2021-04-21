import itertools
import re
import numpy as np

class Monomial:
	def __init__(self):
		self._letters = []

	def append(self, letter):
		if not callable(letter):
			raise TypeError("Cannot append non-callable object")
		self._letters.append(letter)

	def letters(self):
		for letter in self._letters:
			yield letter

	def __call__(self, X):
		X_mul = np.ones(X.shape[1])
		for letter in self.letters():
			X_mul *= letter(X)
		return X_mul

class SummationIterator:
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

	def __call__(self, X):
		return [mon(X) for mon in self.monomials()]

class SimpleWord(SummationIterator):
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
	words = []
	av_elements = [str(i+1) for i in range(dim)]
	for i in range(number):
		length = np.random.randint(1,max_concatenation_length+1)
		conc = ""
		for j in range(length):
			clength = np.random.randint(1,max_letter_weight+1)
			conc += "["+"".join(np.random.choice(av_elements, size=clength))+"]"
		words.append(SimpleWord(conc))
	return words

def generate_words(dim:int=1,
				   monomial_length:int=1,
				   n_monomials:int=1) -> list:
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