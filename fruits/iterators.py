import re
import numpy as np

# nomenclator:
# in a concatination of compositions, [123] is a word
# this word has three letters: 1, 2 and 3
# they can be interpreted as functions

class Word:
	def __init__(self):
		self._letters = []

	def append(self, letter):
		if not callable(letter):
			raise TypeError("Can only append callable functions to variable of \
type Word")
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
	''' abstract class that is given to the iterated_sums function '''
	def __init__(self):
		# array of words
		self._words = []
		self._word_index = 0

	def append(self, word:Word):
		if not isinstance(word, Word):
			raise TypeError("Can only append class type Word to \
SummationIterator")
		self._words.append(word)

	def words(self):
		for word in self._words:
			yield word

	def __call__(self, X):
		return [word(X) for word in self.words()]

	@staticmethod
	def build_from_compositionstring(string:str):
		''' takes a string like [112][1114][223] and returns the corresponding
		SummationIterator '''
		if not re.fullmatch(r"(\[\d+\])+", string):
			return
		words = [x[1:] for x in string.split("]")][:-1]
		iterator = SummationIterator()
		for word in words:
			W = Word()
			counts = {int(letter)-1: word.count(letter) for letter in 
					  set([x for x in word])}
			for i in counts:
				d = {}
				exec(f"def f(X): return X[{i}, :]**{counts[i]}", d)
				W.append(d['f'])
			iterator.append(W)

		return iterator

def generate_concatinations(number:int, dim:int=1,
							max_composition_length:int=5,
							max_concatenation_length:int=10):
	concs = []
	av_elements = [str(i+1) for i in range(dim)]
	for i in range(number):
		length = np.random.randint(1,max_concatenation_length+1)
		conc = ""
		for j in range(length):
			clength = np.random.randint(1,max_composition_length+1)
			conc += "["+"".join(np.random.choice(av_elements, size=clength))+"]"
		concs.append(SummationIterator.build_from_compositionstring(conc))
	return concs