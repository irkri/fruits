"""This python file defines some fruits configurations (fruits.Fruit
objects).
Each configuration aims to answer a part of the big question:

Which configuration is the best one for the classification of time
series data?
"""
import numpy as np

from context import fruits
from complex_words import (
    sigmoid,
    leaky_relu,
    id_,
    tanh,
    generate_complex_words,
    generate_random_complex_words,
    generate_rotated_complex_words,
)

np.random.seed(62)

simple_words_degree_2 = fruits.iterators.generate_words(1, 2, 2)
simple_words_degree_3 = fruits.iterators.generate_words(1, 3, 3)
simple_words_long = simple_words_degree_3[11:36]

# Configurations

# Apple
# Q: What Preparateurs to use?
# A: Probably a mix of ID (nothing) and INC is a good choice.

apple01 = fruits.Fruit("Apple [ID]")
apple01.add(simple_words_degree_3)
apple01.add(fruits.features.PPV(quantile=0.5, sample_size=1))
apple01.add(fruits.features.MAX)
apple01.add(fruits.features.MIN)
apple01.add(fruits.features.END)

apple02 = apple01.deepcopy()
apple02.name = "Apple [STD]"
apple02.add(fruits.preparateurs.STD)

apple03 = apple01.deepcopy()
apple03.name = "Apple [INC]"
apple03.add(fruits.preparateurs.INC)

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
banana01.add(fruits.features.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.features.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.start_new_branch()
banana01.add(fruits.preparateurs.INC)
banana01.add(simple_words_degree_2)
banana01.add(fruits.features.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))
banana01.add(fruits.features.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=False))

banana02 = fruits.Fruit("Banana [Segments]")
banana02.add(simple_words_degree_2)
banana02.add(fruits.features.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.add(fruits.features.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.start_new_branch()
banana02.add(fruits.preparateurs.INC)
banana02.add(simple_words_degree_2)
banana02.add(fruits.features.MAX(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))
banana02.add(fruits.features.MIN(cut=[1,0.2,0.4,0.6,0.8,-1], segments=True))

# Kiwi
# Q: PPV with or without 'segments' option enabled
# A: segments=False seems to work better

kiwi01 = fruits.Fruit("Kiwi")
kiwi01.add(simple_words_degree_2)
kiwi01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                               constant=False,
                               sample_size=1,
                               segments=False))
kiwi01.start_new_branch()
kiwi01.add(fruits.preparateurs.INC)
kiwi01.add(simple_words_degree_2)
kiwi01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                               constant=False,
                               sample_size=1,
                               segments=False))

kiwi02 = fruits.Fruit("Kiwi [Segments]")
kiwi02.add(simple_words_degree_2)
kiwi02.add(fruits.features.PPV([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                               constant=False,
                               sample_size=1,
                               segments=True))
kiwi02.start_new_branch()
kiwi02.add(fruits.preparateurs.INC)
kiwi02.add(simple_words_degree_2)
kiwi02.add(fruits.features.PPV([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                               constant=False,
                               sample_size=1,
                               segments=True))

# Orange
# Q: Which function to choose for complex words, where each letter is
#    the same word?
# A: The pure SimpleWords seem to work best.

orange01 = fruits.Fruit("Orange [id]")
orange01.add(fruits.preparateurs.INC)
orange01.add(simple_words_degree_2)
orange01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange01.add(fruits.features.MAX)
orange01.add(fruits.features.MIN)
orange01.add(fruits.features.END)
orange01.start_new_branch()
orange01.add(simple_words_degree_2)
orange01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange01.add(fruits.features.MAX)
orange01.add(fruits.features.MIN)
orange01.add(fruits.features.END)

orange02 = fruits.Fruit("Orange [leaky_relu]")
orange02.add(fruits.preparateurs.INC)
orange02.add(generate_complex_words(simple_words_degree_2, leaky_relu))
orange02.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange02.add(fruits.features.MAX)
orange02.add(fruits.features.MIN)
orange02.add(fruits.features.END)
orange02.start_new_branch()
orange02.add(generate_complex_words(simple_words_degree_2, leaky_relu))
orange02.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange02.add(fruits.features.MAX)
orange02.add(fruits.features.MIN)
orange02.add(fruits.features.END)

orange03 = fruits.Fruit("Orange [tanh]")
orange03.add(fruits.preparateurs.INC)
orange03.add(generate_complex_words(simple_words_degree_2, tanh))
orange03.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange03.add(fruits.features.MAX)
orange03.add(fruits.features.MIN)
orange03.add(fruits.features.END)
orange03.start_new_branch()
orange03.add(generate_complex_words(simple_words_degree_2, tanh))
orange03.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange03.add(fruits.features.MAX)
orange03.add(fruits.features.MIN)
orange03.add(fruits.features.END)

orange04 = fruits.Fruit("Orange [sigmoid]")
orange04.add(fruits.preparateurs.INC)
orange04.add(generate_complex_words(simple_words_degree_2, sigmoid))
orange04.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange04.add(fruits.features.MAX)
orange04.add(fruits.features.MIN)
orange04.add(fruits.features.END)
orange04.start_new_branch()
orange04.add(generate_complex_words(simple_words_degree_2, sigmoid))
orange04.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
orange04.add(fruits.features.MAX)
orange04.add(fruits.features.MIN)
orange04.add(fruits.features.END)

# Peach
# Q: Are 'rotated' complex words useful? Which functions should we use?
# A: Rotated complex words are useful! Especially if they are created
#    on longer SimpleWords.

peach01 = fruits.Fruit("Peach [id, tanh, sigmoid, leaky_relu]")
peach01.add(fruits.preparateurs.INC)
peach01.add(generate_rotated_complex_words(simple_words_long,
                                           [id_,
                                            tanh,
                                            sigmoid,
                                            leaky_relu]))
peach01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.features.MAX)
peach01.add(fruits.features.MIN)
peach01.add(fruits.features.END)
peach01.start_new_branch()
peach01.add(generate_rotated_complex_words(simple_words_long,
                                           [id_,
                                            tanh,
                                            sigmoid,
                                            leaky_relu]))
peach01.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach01.add(fruits.features.MAX)
peach01.add(fruits.features.MIN)
peach01.add(fruits.features.END)

peach02 = fruits.Fruit("Peach [tanh, sigmoid, leaky_relu]")
peach02.add(fruits.preparateurs.INC)
peach02.add(generate_rotated_complex_words(simple_words_long,
                                           [tanh,
                                            sigmoid,
                                            leaky_relu]))
peach02.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.features.MAX)
peach02.add(fruits.features.MIN)
peach02.add(fruits.features.END)
peach02.start_new_branch()
peach02.add(generate_rotated_complex_words(simple_words_long,
                                           [tanh,
                                            sigmoid,
                                            leaky_relu]))
peach02.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach02.add(fruits.features.MAX)
peach02.add(fruits.features.MIN)
peach02.add(fruits.features.END)

peach03 = fruits.Fruit("Peach [id, leaky_relu]")
peach03.add(fruits.preparateurs.INC)
peach03.add(generate_rotated_complex_words(simple_words_long,
                                           [id_,
                                            leaky_relu]))
peach03.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.features.MAX)
peach03.add(fruits.features.MIN)
peach03.add(fruits.features.END)
peach03.start_new_branch()
peach03.add(generate_rotated_complex_words(simple_words_long,
                                           [id_,
                                            leaky_relu]))
peach03.add(fruits.features.PPV([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                 constant=False,
                                 sample_size=1,
                                 segments=False))
peach03.add(fruits.features.MAX)
peach03.add(fruits.features.MIN)
peach03.add(fruits.features.END)

CONFIGURATIONS = [
    apple01, apple02, apple03, apple04,
    kiwi01, kiwi02,
    banana01, banana02,
    orange01, orange02, orange03, orange04,
    peach01, peach02, peach03,
]
