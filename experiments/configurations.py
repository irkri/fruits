"""This python file defines some fruits configurations (fruits.Fruit
objects).
Each configuration aims to answer a part of the big question:

Which configuration is the best one for the classification of time
series data?
"""
import numpy as np

from context import fruits
from fruits.core.generation import replace_letters

np.random.seed(62)

# word definitions

simple_words_degree_2 = fruits.core.generation.simplewords_by_degree(2, 2, 1)
simple_words_degree_3 = fruits.core.generation.simplewords_by_degree(3, 3, 1)
simple_words_long = fruits.core.generation.simplewords_by_weight(4, 1)

# complex letter definitions

@fruits.core.complex_letter(name="SIGMOID")
def sigmoid(X: np.ndarray, i: int):
    return 1 / (1 + np.exp(-0.001*X[i, :]))

@fruits.core.complex_letter(name="leakyRELU")
def leaky_relu(X: np.ndarray, i: int):
    out = np.zeros(X.shape[1], dtype=np.float64)
    out += X[i, :] * (X[i, :]>0)
    out += (X[i, :]*0.005) * (X[i, :]<=0)
    return out

@fruits.core.complex_letter(name="TANH")
def tanh(X: np.ndarray, i: int):
    pos = np.exp(0.001*X[i, :])
    neg = np.exp(-0.001*X[i, :])
    return (pos-neg) / (pos+neg)

@fruits.core.complex_letter(name="ID")
def id_(X: np.ndarray, i: int):
    return X[i, :]

# Configurations

# Apple
# Q: What Preparateurs to use?
# A: Probably a mix of ID (nothing) and INC is a good choice.

apple01 = fruits.Fruit("Apple [ID]")
apple01.add(simple_words_degree_3)
apple01.add(fruits.sieving.PPV(quantile=0.5, sample_size=1))
apple01.add(fruits.sieving.MAX)
apple01.add(fruits.sieving.MIN)
apple01.add(fruits.sieving.END)

apple02 = apple01.deepcopy()
apple02.name = "Apple [STD]"
apple02.add(fruits.preparation.STD)

apple03 = apple01.deepcopy()
apple03.name = "Apple [INC]"
apple03.add(fruits.preparation.INC)

apple04 = apple01.deepcopy()
apple04.name = "Apple [ID + INC]"
apple04.fork(apple03.branch().deepcopy())

# Banana
# Q: MAX/MIN with 'segments' True or False
# A: Mean accuracy with segments is better but the more important,
#    bigger datasets classify better with 'segments' disabled.

banana01 = fruits.Fruit("Banana")
banana01.add(simple_words_degree_2)
banana01.add(fruits.sieving.MAX(cut=[0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.sieving.MIN(cut=[0.2,0.4,0.6,0.8,-1], segments=False))
banana01.fork()
banana01.add(fruits.preparation.INC)
banana01.add(simple_words_degree_2)
banana01.add(fruits.sieving.MAX(cut=[0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.sieving.MIN(cut=[0.2,0.4,0.6,0.8,-1], segments=False))

banana02 = fruits.Fruit("Banana [Segments]")
banana02.add(simple_words_degree_2)
banana02.add(fruits.sieving.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.add(fruits.sieving.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.fork()
banana02.add(fruits.preparation.INC)
banana02.add(simple_words_degree_2)
banana02.add(fruits.sieving.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.add(fruits.sieving.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))

# Kiwi
# Q: PPV with or without 'segments' option enabled
# A: segments=False seems to work better

kiwi01 = fruits.Fruit("Kiwi")
kiwi01.add(simple_words_degree_2)
kiwi01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                               constant=False,
                               sample_size=1,
                               segments=False))
kiwi01.fork()
kiwi01.add(fruits.preparation.INC)
kiwi01.add(simple_words_degree_2)
kiwi01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                               constant=False,
                               sample_size=1,
                               segments=False))

kiwi02 = fruits.Fruit("Kiwi [Segments]")
kiwi02.add(simple_words_degree_2)
kiwi02.add(fruits.sieving.PPV([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                               constant=False,
                               sample_size=1,
                               segments=True))
kiwi02.fork()
kiwi02.add(fruits.preparation.INC)
kiwi02.add(simple_words_degree_2)
kiwi02.add(fruits.sieving.PPV([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                               constant=False,
                               sample_size=1,
                               segments=True))

# Peach
# Q: Are complex words with sequentially replaced letters useful?
#    Which letters should we use?
# A: Longer words lead to better results.

def letter_gen(letters: list):
    i = 0
    while i < 1000:
        yield letters[i % len(letters)]
        i += 1

peach01 = fruits.Fruit("Peach [id, tanh, sigmoid, leaky_relu]")
peach01.add(fruits.preparation.INC)
peach01.add(replace_letters(simple_words_long,
                            letter_gen([id_, tanh, sigmoid, leaky_relu])))
peach01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.sieving.MAX)
peach01.add(fruits.sieving.MIN)
peach01.add(fruits.sieving.END)
peach01.fork()
peach01.add(replace_letters(simple_words_long,
                            letter_gen([id_, tanh, sigmoid, leaky_relu])))
peach01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.sieving.MAX)
peach01.add(fruits.sieving.MIN)
peach01.add(fruits.sieving.END)

peach02 = fruits.Fruit("Peach [tanh, sigmoid, leaky_relu]")
peach02.add(fruits.preparation.INC)
peach02.add(replace_letters(simple_words_long,
                            letter_gen([tanh, sigmoid, leaky_relu])))
peach02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.sieving.MAX)
peach02.add(fruits.sieving.MIN)
peach02.add(fruits.sieving.END)
peach02.fork()
peach02.add(replace_letters(simple_words_long,
                            letter_gen([tanh, sigmoid, leaky_relu])))
peach02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.sieving.MAX)
peach02.add(fruits.sieving.MIN)
peach02.add(fruits.sieving.END)

peach03 = fruits.Fruit("Peach [id, leaky_relu]")
peach03.add(fruits.preparation.INC)
peach03.add(replace_letters(simple_words_long,
                            letter_gen([id_, leaky_relu])))
peach03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.sieving.MAX)
peach03.add(fruits.sieving.MIN)
peach03.add(fruits.sieving.END)
peach03.fork()
peach03.add(replace_letters(simple_words_long,
                            letter_gen([id_, leaky_relu])))
peach03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.sieving.MAX)
peach03.add(fruits.sieving.MIN)
peach03.add(fruits.sieving.END)

CONFIGURATIONS = [
    apple01, apple02, apple03, apple04,
    kiwi01, kiwi02,
    banana01, banana02,
    peach01, peach02, peach03,
]
