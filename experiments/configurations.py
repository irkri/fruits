"""This python file defines some fruits configurations (fruits.Fruit
objects).
Each configuration aims to answer a part of the big question:

Which configuration is the best one for the classification of time
series data?
"""
import numpy as np

from context import fruits
from fruits.core.generation import simplewords_replace_letters
from complex_words import (
    sigmoid,
    leaky_relu,
    id_,
    tanh,
    simplewords_replace_letters_randomly,
    simplewords_replace_letters_sequentially,
)

np.random.seed(62)

simple_words_degree_2 = fruits.core.generation.simplewords_by_degree(2, 2, 1)
simple_words_degree_3 = fruits.core.generation.simplewords_by_degree(3, 3, 1)
simple_words_long = fruits.core.generation.simplewords_by_length(4, 1)

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
apple04.add_branch(apple03.current_branch().deepcopy())

# Banana
# Q: MAX/MIN with 'segments' True or False
# A: Mean accuracy with segments is better but the more important,
#    bigger datasets classify better with 'segments' disabled.
#    BUT: segments=True also produces less features
#    (1 per sieve and iterator)

banana01 = fruits.Fruit("Banana")
banana01.add(simple_words_degree_2)
banana01.add(fruits.sieving.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.sieving.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.start_new_branch()
banana01.add(fruits.preparation.INC)
banana01.add(simple_words_degree_2)
banana01.add(fruits.sieving.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.sieving.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))

banana02 = fruits.Fruit("Banana [Segments]")
banana02.add(simple_words_degree_2)
banana02.add(fruits.sieving.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.add(fruits.sieving.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.start_new_branch()
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
kiwi01.start_new_branch()
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
kiwi02.start_new_branch()
kiwi02.add(fruits.preparation.INC)
kiwi02.add(simple_words_degree_2)
kiwi02.add(fruits.sieving.PPV([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                               constant=False,
                               sample_size=1,
                               segments=True))

# Orange
# Q: Which function to choose for complex words, where each letter is
#    the same word?
# A: The pure SimpleWords seem to work best.

orange01 = fruits.Fruit("Orange [id]")
orange01.add(fruits.preparation.INC)
orange01.add(simple_words_degree_2)
orange01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange01.add(fruits.sieving.MAX)
orange01.add(fruits.sieving.MIN)
orange01.add(fruits.sieving.END)
orange01.start_new_branch()
orange01.add(simple_words_degree_2)
orange01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange01.add(fruits.sieving.MAX)
orange01.add(fruits.sieving.MIN)
orange01.add(fruits.sieving.END)

orange02 = fruits.Fruit("Orange [leaky_relu]")
orange02.add(fruits.preparation.INC)
orange02.add(simplewords_replace_letters(simple_words_degree_2, leaky_relu))
orange02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange02.add(fruits.sieving.MAX)
orange02.add(fruits.sieving.MIN)
orange02.add(fruits.sieving.END)
orange02.start_new_branch()
orange02.add(simplewords_replace_letters(simple_words_degree_2, leaky_relu))
orange02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange02.add(fruits.sieving.MAX)
orange02.add(fruits.sieving.MIN)
orange02.add(fruits.sieving.END)

orange03 = fruits.Fruit("Orange [tanh]")
orange03.add(fruits.preparation.INC)
orange03.add(simplewords_replace_letters(simple_words_degree_2, tanh))
orange03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange03.add(fruits.sieving.MAX)
orange03.add(fruits.sieving.MIN)
orange03.add(fruits.sieving.END)
orange03.start_new_branch()
orange03.add(simplewords_replace_letters(simple_words_degree_2, tanh))
orange03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange03.add(fruits.sieving.MAX)
orange03.add(fruits.sieving.MIN)
orange03.add(fruits.sieving.END)

orange04 = fruits.Fruit("Orange [sigmoid]")
orange04.add(fruits.preparation.INC)
orange04.add(simplewords_replace_letters(simple_words_degree_2, sigmoid))
orange04.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange04.add(fruits.sieving.MAX)
orange04.add(fruits.sieving.MIN)
orange04.add(fruits.sieving.END)
orange04.start_new_branch()
orange04.add(simplewords_replace_letters(simple_words_degree_2, sigmoid))
orange04.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange04.add(fruits.sieving.MAX)
orange04.add(fruits.sieving.MIN)
orange04.add(fruits.sieving.END)

# Peach
# Q: Are 'rotated' complex words useful? Which functions should we use?
# A: Rotated complex words are useful! Especially if they are created
#    on longer SimpleWords.

peach01 = fruits.Fruit("Peach [id, tanh, sigmoid, leaky_relu]")
peach01.add(fruits.preparation.INC)
peach01.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [id_,
                                            tanh,
                                            sigmoid,
                                            leaky_relu]))
peach01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.sieving.MAX)
peach01.add(fruits.sieving.MIN)
peach01.add(fruits.sieving.END)
peach01.start_new_branch()
peach01.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [id_,
                                            tanh,
                                            sigmoid,
                                            leaky_relu]))
peach01.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.sieving.MAX)
peach01.add(fruits.sieving.MIN)
peach01.add(fruits.sieving.END)

peach02 = fruits.Fruit("Peach [tanh, sigmoid, leaky_relu]")
peach02.add(fruits.preparation.INC)
peach02.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [tanh,
                                            sigmoid,
                                            leaky_relu]))
peach02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.sieving.MAX)
peach02.add(fruits.sieving.MIN)
peach02.add(fruits.sieving.END)
peach02.start_new_branch()
peach02.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [tanh,
                                            sigmoid,
                                            leaky_relu]))
peach02.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.sieving.MAX)
peach02.add(fruits.sieving.MIN)
peach02.add(fruits.sieving.END)

peach03 = fruits.Fruit("Peach [id, leaky_relu]")
peach03.add(fruits.preparation.INC)
peach03.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [id_,
                                            leaky_relu]))
peach03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.sieving.MAX)
peach03.add(fruits.sieving.MIN)
peach03.add(fruits.sieving.END)
peach03.start_new_branch()
peach03.add(simplewords_replace_letters_sequentially(simple_words_long,
                                           [id_,
                                            leaky_relu]))
peach03.add(fruits.sieving.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.sieving.MAX)
peach03.add(fruits.sieving.MIN)
peach03.add(fruits.sieving.END)

CONFIGURATIONS = [
    # apple01, apple02, apple03, apple04,
    # kiwi01, kiwi02,
    # banana01, banana02,
    # orange01, orange02, orange03, orange04,
    peach01, peach02, peach03,
]
