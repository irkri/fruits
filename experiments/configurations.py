from context import fruits
import numpy as np

# function definitions for more complicated words
def SIGMOID(i):
	def sigmoid(X):
		return 1/(1+np.exp(-X[i, :]))
	return sigmoid
def RELU(i):
	def relu(X):
		return X[i, :]*(X[i, :]>0)
	return relu
def leakyRELU(i, alpha=0.01):
	def leakyrelu(X):
		out = X[i, :]*(X[i, :]>0)
		out += ((X[i, :]<=0)*X[i, :]*alpha)
		return out
	return leakyrelu
def SINE(i):
	def sin(X):
		return np.sin(X[i, :])
	return sin
def IEXP(i):
	def iexp(X):
		return np.exp(X[i, :])
	return iexp
def TANH(i):
	def tanh(X):
		pos = np.exp(X[i, :])
		neg = np.exp(-X[i, :])
		return (pos-neg)/(pos+neg)
	return tanh

def generate_complex_words(simple_words, FUNCTION, scale:int=0):
	complex_words = []
	for simple_word in simple_words:
		complex_words.append(fruits.iterators.SummationIterator(
												str(simple_word)[11:-1]))
		for monomial in simple_word.monomials():
			mon = []
			for i, letter in enumerate(monomial):
				for l in range(letter):
					mon.append(FUNCTION(i))
		complex_words[-1].append(mon)
		complex_words[-1].scale = scale
	return complex_words

simple_words_degree_3 = fruits.iterators.generate_words(1, 3, 3)
simple_words_degree_4 = fruits.iterators.generate_words(1, 4, 4)

# configuration 1
apple = fruits.Fruit("Apple")
apple.add(fruits.preparateurs.INC)

apple.add(simple_words_degree_4)

apple.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
apple.add(fruits.features.MAX)
apple.add(fruits.features.MIN)

# configuration 2
banana = fruits.Fruit("Banana")
banana.add(fruits.preparateurs.STD)

banana.add(simple_words_degree_4)

banana.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
banana.add(fruits.features.MAX)
banana.add(fruits.features.MIN)

# configuration 3
orange = fruits.Fruit("Orange")
orange.add(fruits.preparateurs.INC)

orange.add(simple_words_degree_4)

orange.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
orange.add(fruits.features.MAX)
orange.add(fruits.features.MIN)
orange.add(fruits.features.END)

# configuration 4
peach = fruits.Fruit("Peach")

peach.add(simple_words_degree_4)

peach.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
peach.add(fruits.features.MAX)
peach.add(fruits.features.MIN)
peach.add(fruits.features.END)

#configuration 5
watermelon = fruits.Fruit("Watermelon")

watermelon.add(generate_complex_words(simple_words_degree_4, TANH))

watermelon.add(fruits.features.PPV(quantile=0.2, constant=False, sample_size=1))
watermelon.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
watermelon.add(fruits.features.PPV(quantile=0.8, constant=False, sample_size=1))
watermelon.add(fruits.features.MAX)
watermelon.add(fruits.features.MIN)
watermelon.add(fruits.features.END)

# configuration 6
strawberry = fruits.Fruit("Strawberry")

strawberry.add(generate_complex_words(simple_words_degree_4, leakyRELU))

strawberry.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
strawberry.add(fruits.features.MAX)
strawberry.add(fruits.features.MIN)
strawberry.add(fruits.features.END)

# configuration 7
pineapple = fruits.Fruit("Pineapple")

pineapple.add(generate_complex_words(simple_words_degree_3, RELU))
pineapple.add(generate_complex_words(simple_words_degree_3, SIGMOID))
pineapple.add(simple_words_degree_3)

pineapple.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
pineapple.add(fruits.features.MAX)
pineapple.add(fruits.features.MIN)
pineapple.add(fruits.features.END)

# configuration 8
cranberry = fruits.Fruit("Cranberry")
cranberry.add(fruits.preparateurs.INC)

cranberry.add(generate_complex_words(simple_words_degree_3, IEXP, scale=3))
cranberry.add(simple_words_degree_3)

cranberry.add(fruits.features.MAX)
cranberry.add(fruits.features.MIN)

# configuration 9
blackberry = fruits.Fruit("Blackberry")
blackberry.add(fruits.preparateurs.INC)

blackberry.add(generate_complex_words(simple_words_degree_3, RELU))
blackberry.add(generate_complex_words(simple_words_degree_3, SIGMOID))
blackberry.add(fruits.iterators.generate_words(1, 5, 3))

blackberry.add(fruits.features.PPV(quantile=0.1, constant=False, sample_size=1))
blackberry.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
blackberry.add(fruits.features.PPV(quantile=0.9, constant=False, sample_size=1))
blackberry.add(fruits.features.MAX)
blackberry.add(fruits.features.MIN)
blackberry.add(fruits.features.END)

# configuration 10
starfruit = fruits.Fruit("Starfruit")
starfruit.add(fruits.preparateurs.INC)
starfruit.add(simple_words_degree_3)
starfruit.add(fruits.features.PPV(quantile=0.2, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.8, constant=False, sample_size=1))
starfruit.add(fruits.features.MAX)
starfruit.add(fruits.features.MIN)
starfruit.add(fruits.features.END)

starfruit.start_new_branch()
starfruit.add(simple_words_degree_3)
starfruit.add(fruits.features.PPV(quantile=0.2, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.8, constant=False, sample_size=1))
starfruit.add(fruits.features.MAX)
starfruit.add(fruits.features.MIN)
starfruit.add(fruits.features.END)

starfruit.start_new_branch()
starfruit.add(generate_complex_words(simple_words_degree_3, leakyRELU))
starfruit.add(fruits.features.PPV(quantile=0.2, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.5, constant=False, sample_size=1))
starfruit.add(fruits.features.PPV(quantile=0.8, constant=False, sample_size=1))
starfruit.add(fruits.features.MAX)
starfruit.add(fruits.features.MIN)
starfruit.add(fruits.features.END)

CONFIGURATIONS = [apple,
				  banana,
				  orange,
				  peach,
				  watermelon,
				  strawberry,
				  pineapple,
				  cranberry,
				  blackberry,
				  starfruit]
