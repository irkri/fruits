"""This python module defines some fruits to use for feature extraction.
The dictionary ``basket`` contains grouped experiments where one element
is a tuple of fruits all with small changes to compare different
settings within a fruit.
"""

__all__ = ["basket"]

import fruits
import numpy as np

np.random.seed(62)

# Configuration 00 - Grape - INC preparateur

words = fruits.words.of_weight(4)

grape01 = fruits.Fruit("Grape_without")
grape01.add(*words)
grape01.branch().configure(mode="extended")
grape01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

grape02 = fruits.Fruit("Grape_with")
grape02.add(fruits.preparation.INC)
grape02.add(*words)
grape02.branch().configure(mode="extended")
grape02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

grape03 = fruits.Fruit("Grape_both")
grape03.add(fruits.preparation.INC)
grape03.add(*words)
grape03.branch().configure(mode="extended")
grape03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

grape03.fork()
grape03.add(*words)
grape03.branch().configure(mode="extended")
grape03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

# Configuration 01 - Apple - Number of words

words = fruits.words.of_weight(3)

apple01 = fruits.Fruit("Apple_3")
apple01.add(fruits.preparation.INC)
apple01.add(*words)
apple01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apple01.fork()
apple01.add(*words)
apple01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

apple02 = fruits.Fruit("Apple_4")
apple02.add(fruits.preparation.INC)
apple02.add(*words)
apple02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apple02.fork()
apple02.add(*words)
apple02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

apple03 = fruits.Fruit("Apple_1_4")
apple03.add(fruits.preparation.INC)
apple03.add(*words)
apple03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)
apple03.branch().configure(mode="extended")

apple03.fork()
apple03.add(*words)
apple03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)
apple03.branch().configure(mode="extended")

words = fruits.words.of_weight(5)

apple04 = fruits.Fruit("Apple_1_5")
apple04.add(fruits.preparation.INC)
apple04.add(*words)
apple04.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)
apple04.branch().configure(mode="extended")

apple04.fork()
apple04.add(*words)
apple04.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)
apple04.branch().configure(mode="extended")

# Configuration 02 - Banana - Number of PPV Quantiles

words = fruits.words.of_weight(4)

banana01 = fruits.Fruit("Banana_1")
banana01.add(fruits.preparation.INC)
banana01.add(*words)
banana01.branch().configure(mode="extended")
banana01.add(fruits.sieving.PPV(0.5))

banana01.fork()
banana01.add(*words)
banana01.branch().configure(mode="extended")
banana01.add(fruits.sieving.PPV(0.5))

banana02 = fruits.Fruit("Banana_3")
banana02.add(fruits.preparation.INC)
banana02.add(*words)
banana02.branch().configure(mode="extended")
banana02.add(fruits.sieving.PPV([0.25, 0.5, 0.75]))

banana02.fork()
banana02.add(*words)
banana02.branch().configure(mode="extended")
banana02.add(fruits.sieving.PPV([0.25, 0.5, 0.75]))

banana03 = fruits.Fruit("Banana_5")
banana03.add(fruits.preparation.INC)
banana03.add(*words)
banana03.branch().configure(mode="extended")
banana03.add(fruits.sieving.PPV([i/6 for i in range(1, 6)]))

banana03.fork()
banana03.add(*words)
banana03.branch().configure(mode="extended")
banana03.add(fruits.sieving.PPV([i/6 for i in range(1, 6)]))

banana04 = fruits.Fruit("Banana_7")
banana04.add(fruits.preparation.INC)
banana04.add(*words)
banana04.branch().configure(mode="extended")
banana04.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))

banana04.fork()
banana04.add(*words)
banana04.branch().configure(mode="extended")
banana04.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))

banana05 = fruits.Fruit("Banana_9")
banana05.add(fruits.preparation.INC)
banana05.add(*words)
banana05.branch().configure(mode="extended")
banana05.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))

banana05.fork()
banana05.add(*words)
banana05.branch().configure(mode="extended")
banana05.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))

banana06 = fruits.Fruit("Banana_19")
banana06.add(fruits.preparation.INC)
banana06.add(*words)
banana06.branch().configure(mode="extended")
banana06.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))

banana06.fork()
banana06.add(*words)
banana06.branch().configure(mode="extended")
banana06.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))

# Configuration 03 - Plantain - Number of CPV Quantiles

words = fruits.words.of_weight(4)

plantain01 = fruits.Fruit("Plantain_1")
plantain01.add(fruits.preparation.INC)
plantain01.add(*words)
plantain01.branch().configure(mode="extended")
plantain01.add(fruits.sieving.CPV(0.5))

plantain01.fork()
plantain01.add(*words)
plantain01.branch().configure(mode="extended")
plantain01.add(fruits.sieving.CPV(0.5))

plantain02 = fruits.Fruit("Plantain_3")
plantain02.add(fruits.preparation.INC)
plantain02.add(*words)
plantain02.branch().configure(mode="extended")
plantain02.add(fruits.sieving.CPV([0.25, 0.5, 0.75]))

plantain02.fork()
plantain02.add(*words)
plantain02.branch().configure(mode="extended")
plantain02.add(fruits.sieving.CPV([0.25, 0.5, 0.75]))

plantain03 = fruits.Fruit("Plantain_5")
plantain03.add(fruits.preparation.INC)
plantain03.add(*words)
plantain03.branch().configure(mode="extended")
plantain03.add(fruits.sieving.CPV([i/6 for i in range(1, 6)]))

plantain03.fork()
plantain03.add(*words)
plantain03.branch().configure(mode="extended")
plantain03.add(fruits.sieving.CPV([i/6 for i in range(1, 6)]))

plantain04 = fruits.Fruit("Plantain_7")
plantain04.add(fruits.preparation.INC)
plantain04.add(*words)
plantain04.branch().configure(mode="extended")
plantain04.add(fruits.sieving.CPV([i/8 for i in range(1, 8)]))

plantain04.fork()
plantain04.add(*words)
plantain04.branch().configure(mode="extended")
plantain04.add(fruits.sieving.CPV([i/8 for i in range(1, 8)]))

plantain05 = fruits.Fruit("Plantain_9")
plantain05.add(fruits.preparation.INC)
plantain05.add(*words)
plantain05.branch().configure(mode="extended")
plantain05.add(fruits.sieving.CPV([i/10 for i in range(1, 10)]))

plantain05.fork()
plantain05.add(*words)
plantain05.branch().configure(mode="extended")
plantain05.add(fruits.sieving.CPV([i/10 for i in range(1, 10)]))

plantain06 = fruits.Fruit("Plantain_19")
plantain06.add(fruits.preparation.INC)
plantain06.add(*words)
plantain06.branch().configure(mode="extended")
plantain06.add(fruits.sieving.CPV([i/20 for i in range(1, 20)]))

plantain06.fork()
plantain06.add(*words)
plantain06.branch().configure(mode="extended")
plantain06.add(fruits.sieving.CPV([i/20 for i in range(1, 20)]))

# Configuration 04 - Orange - Number of cuts in MAX

words = fruits.words.of_weight(4)

orange01 = fruits.Fruit("Orange_2")
orange01.add(fruits.preparation.INC)
orange01.add(*words)
orange01.branch().configure(mode="extended")
orange01.add(fruits.sieving.MAX([1, 0.5, -1]))

orange01.fork()
orange01.add(*words)
orange01.branch().configure(mode="extended")
orange01.add(fruits.sieving.MAX([1, 0.5, -1]))

orange02 = fruits.Fruit("Orange_5")
orange02.add(fruits.preparation.INC)
orange02.add(*words)
orange02.branch().configure(mode="extended")
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1]))

orange02.fork()
orange02.add(*words)
orange02.branch().configure(mode="extended")
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1]))

orange03 = fruits.Fruit("Orange_10")
orange03.add(fruits.preparation.INC)
orange03.add(*words)
orange03.branch().configure(mode="extended")
orange03.add(fruits.sieving.MAX([i/10 for i in range(0, 11)]))

orange03.fork()
orange03.add(*words)
orange03.branch().configure(mode="extended")
orange03.add(fruits.sieving.MAX([i/10 for i in range(0, 11)]))

orange04 = fruits.Fruit("Orange_20")
orange04.add(fruits.preparation.INC)
orange04.add(*words)
orange04.branch().configure(mode="extended")
orange04.add(fruits.sieving.MAX([i/20 for i in range(0, 21)]))

orange04.fork()
orange04.add(*words)
orange04.branch().configure(mode="extended")
orange04.add(fruits.sieving.MAX([i/20 for i in range(0, 21)]))

# Configuration 05 - Tangerine - Number of cuts in MIN

words = fruits.words.of_weight(4)

tangerine01 = fruits.Fruit("Tangerine_2")
tangerine01.add(fruits.preparation.INC)
tangerine01.add(*words)
tangerine01.branch().configure(mode="extended")
tangerine01.add(fruits.sieving.MIN([1, 0.5, -1]))

tangerine01.fork()
tangerine01.add(*words)
tangerine01.branch().configure(mode="extended")
tangerine01.add(fruits.sieving.MIN([1, 0.5, -1]))

tangerine02 = fruits.Fruit("Tangerine_5")
tangerine02.add(fruits.preparation.INC)
tangerine02.add(*words)
tangerine02.branch().configure(mode="extended")
tangerine02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1]))

tangerine02.fork()
tangerine02.add(*words)
tangerine02.branch().configure(mode="extended")
tangerine02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1]))

tangerine03 = fruits.Fruit("Tangerine_10")
tangerine03.add(fruits.preparation.INC)
tangerine03.add(*words)
tangerine03.branch().configure(mode="extended")
tangerine03.add(fruits.sieving.MIN([i/10 for i in range(0, 11)]))

tangerine03.fork()
tangerine03.add(*words)
tangerine03.branch().configure(mode="extended")
tangerine03.add(fruits.sieving.MIN([i/10 for i in range(0, 11)]))

tangerine04 = fruits.Fruit("Tangerine_20")
tangerine04.add(fruits.preparation.INC)
tangerine04.add(*words)
tangerine04.branch().configure(mode="extended")
tangerine04.add(fruits.sieving.MIN([i/20 for i in range(0, 21)]))

tangerine04.fork()
tangerine04.add(*words)
tangerine04.branch().configure(mode="extended")
tangerine04.add(fruits.sieving.MIN([i/20 for i in range(0, 21)]))

# Configuration 06 - Apricot - Alpha value of words

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = 0.0001
apricot01 = fruits.Fruit("Apricot_1")
apricot01.add(fruits.preparation.INC)
apricot01.add(*words)
apricot01.branch().configure(mode="extended")
apricot01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot01.fork()
apricot01.add(*words)
apricot01.branch().configure(mode="extended")
apricot01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = -0.0001
apricot02 = fruits.Fruit("Apricot_m1")
apricot02.add(fruits.preparation.INC)
apricot02.add(*words)
apricot02.branch().configure(mode="extended")
apricot02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot02.fork()
apricot02.add(*words)
apricot02.branch().configure(mode="extended")
apricot02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = 0.0005
apricot03 = fruits.Fruit("Apricot_5")
apricot03.add(fruits.preparation.INC)
apricot03.add(*words)
apricot03.branch().configure(mode="extended")
apricot03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot03.fork()
apricot03.add(*words)
apricot03.branch().configure(mode="extended")
apricot03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = -0.0005
apricot04 = fruits.Fruit("Apricot_m5")
apricot04.add(fruits.preparation.INC)
apricot04.add(*words)
apricot04.branch().configure(mode="extended")
apricot04.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot04.fork()
apricot04.add(*words)
apricot04.branch().configure(mode="extended")
apricot04.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = 0.001
apricot05 = fruits.Fruit("Apricot_10")
apricot05.add(fruits.preparation.INC)
apricot05.add(*words)
apricot05.branch().configure(mode="extended")
apricot05.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot05.fork()
apricot05.add(*words)
apricot05.branch().configure(mode="extended")
apricot05.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = -0.001
apricot06 = fruits.Fruit("Apricot_m10")
apricot06.add(fruits.preparation.INC)
apricot06.add(*words)
apricot06.branch().configure(mode="extended")
apricot06.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot06.fork()
apricot06.add(*words)
apricot06.branch().configure(mode="extended")
apricot06.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = 0.005
apricot07 = fruits.Fruit("Apricot_50")
apricot07.add(fruits.preparation.INC)
apricot07.add(*words)
apricot07.branch().configure(mode="extended")
apricot07.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot07.fork()
apricot07.add(*words)
apricot07.branch().configure(mode="extended")
apricot07.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(4)

for word in words:
    word.alpha = -0.005
apricot08 = fruits.Fruit("Apricot_m50")
apricot08.add(fruits.preparation.INC)
apricot08.add(*words)
apricot08.branch().configure(mode="extended")
apricot08.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

apricot08.fork()
apricot08.add(*words)
apricot08.branch().configure(mode="extended")
apricot08.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

# Configuration 07 - Olive - Number of weighted words + ONE preparateur

words = [
    fruits.words.SimpleWord("[1]"),
    fruits.words.SimpleWord("[1][2]"),
    fruits.words.SimpleWord("[2][1]"),
]
for word in words:
    word.alpha = -0.005

olive01 = fruits.Fruit("Olive_2")
olive01.add(fruits.preparation.INC)
olive01.add(fruits.preparation.ONE)
olive01.add(*words)
olive01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

olive01.fork()
olive01.add(fruits.preparation.ONE)
olive01.add(*words)
olive01.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = [
    fruits.words.SimpleWord("[1]"),
    fruits.words.SimpleWord("[1][2]"),
    fruits.words.SimpleWord("[2][1][2]"),
    fruits.words.SimpleWord("[1][2][2]"),
    fruits.words.SimpleWord("[2][1][1]"),
    fruits.words.SimpleWord("[2][2][1]"),
    fruits.words.SimpleWord("[1][1][2]"),
]
for word in words:
    word.alpha = -0.005

olive02 = fruits.Fruit("Olive_3")
olive02.add(fruits.preparation.INC)
olive02.add(fruits.preparation.ONE)
olive02.add(*words)
olive02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

olive02.fork()
olive02.add(fruits.preparation.ONE)
olive02.add(*words)
olive02.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

words = fruits.words.of_weight(3, 2)
for word in words:
    word.alpha = -0.005

olive03 = fruits.Fruit("Olive_3all")
olive03.add(fruits.preparation.INC)
olive03.add(fruits.preparation.ONE)
olive03.add(*words)
olive03.branch().configure(mode="extended")
olive03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

olive03.fork()
olive03.add(fruits.preparation.ONE)
olive03.add(*words)
olive03.branch().configure(mode="extended")
olive03.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

# Configuration 08 - Elderberry - Number of coquantiles in END

words = fruits.words.of_weight(4)

elderberry01 = fruits.Fruit("Elderberry_1")
elderberry01.add(fruits.preparation.INC)
elderberry01.add(*words)
elderberry01.branch().configure(mode="extended")
elderberry01.add(fruits.sieving.END([-1]))

elderberry01.fork()
elderberry01.add(*words)
elderberry01.branch().configure(mode="extended")
elderberry01.add(fruits.sieving.END([-1]))

elderberry02 = fruits.Fruit("Elderberry_2")
elderberry02.add(fruits.preparation.INC)
elderberry02.add(*words)
elderberry02.branch().configure(mode="extended")
elderberry02.add(fruits.sieving.END([0.5, -1]))

elderberry02.fork()
elderberry02.add(*words)
elderberry02.branch().configure(mode="extended")
elderberry02.add(fruits.sieving.END([0.5, -1]))

elderberry03 = fruits.Fruit("Elderberry_5")
elderberry03.add(fruits.preparation.INC)
elderberry03.add(*words)
elderberry03.branch().configure(mode="extended")
elderberry03.add(fruits.sieving.END([0.2, 0.4, 0.6, 0.8, -1]))

elderberry03.fork()
elderberry03.add(*words)
elderberry03.branch().configure(mode="extended")
elderberry03.add(fruits.sieving.END([0.2, 0.4, 0.6, 0.8, -1]))

elderberry04 = fruits.Fruit("Elderberry_10")
elderberry04.add(fruits.preparation.INC)
elderberry04.add(*words)
elderberry04.branch().configure(mode="extended")
elderberry04.add(fruits.sieving.END([i/10 for i in range(1, 11)]))

elderberry04.fork()
elderberry04.add(*words)
elderberry04.branch().configure(mode="extended")
elderberry04.add(fruits.sieving.END([i/10 for i in range(1, 11)]))

elderberry05 = fruits.Fruit("Elderberry_20")
elderberry05.add(fruits.preparation.INC)
elderberry05.add(*words)
elderberry05.branch().configure(mode="extended")
elderberry05.add(fruits.sieving.END([i/20 for i in range(1, 21)]))

elderberry05.fork()
elderberry05.add(*words)
elderberry05.branch().configure(mode="extended")
elderberry05.add(fruits.sieving.END([i/20 for i in range(1, 21)]))

# Configuration 09 - Dragonfruit - DIL, WIN or DOT preparateur

words = fruits.words.of_weight(4)

dragonfruit01 = fruits.Fruit("Dragonfruit_DIL")
for n in range(20):
    dragonfruit01.fork()
    dragonfruit01.add(fruits.preparation.INC)
    dragonfruit01.add(fruits.preparation.DIL)
    dragonfruit01.add(*words)
    dragonfruit01.branch().configure(mode="extended")
    dragonfruit01.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

    dragonfruit01.fork()
    dragonfruit01.add(fruits.preparation.DIL)
    dragonfruit01.add(*words)
    dragonfruit01.branch().configure(mode="extended")
    dragonfruit01.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

cuts = [
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.0, 0.2),
    (0.1, 0.3),
    (0.2, 0.4),
    (0.3, 0.5),
    (0.4, 0.6),
    (0.5, 0.7),
    (0.6, 0.8),
    (0.7, 0.9),
    (0.8, 1.0),
    (0.0, 0.5),
    (0.25, 0.75),
    (0.5, 1.0),
]

dragonfruit02 = fruits.Fruit("Dragonfruit_WIN")
for left, right in cuts:
    dragonfruit02.fork()
    dragonfruit02.add(fruits.preparation.INC)
    dragonfruit02.add(fruits.preparation.WIN(left, right))
    dragonfruit02.add(*words)
    dragonfruit02.branch().configure(mode="extended")
    dragonfruit02.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

    dragonfruit02.fork()
    dragonfruit02.add(fruits.preparation.WIN(left, right))
    dragonfruit02.add(*words)
    dragonfruit02.branch().configure(mode="extended")
    dragonfruit02.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

ns = [2, 5, 10]
ns += [i/100 for i in list(range(1, 11))+[15, 20, 25, 30, 35, 40, 45]]

dragonfruit03 = fruits.Fruit("Dragonfruit_DOT")
for n in ns:
    dragonfruit03.fork()
    dragonfruit03.add(fruits.preparation.INC)
    dragonfruit03.add(fruits.preparation.DOT(n))
    dragonfruit03.add(*words)
    dragonfruit03.branch().configure(mode="extended")
    dragonfruit03.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

    dragonfruit03.fork()
    dragonfruit03.add(fruits.preparation.DOT(n))
    dragonfruit03.add(*words)
    dragonfruit03.branch().configure(mode="extended")
    dragonfruit03.add(
        fruits.sieving.PPV,
        fruits.sieving.CPV,
        fruits.sieving.PIA,
        fruits.sieving.MAX,
        fruits.sieving.MIN,
        fruits.sieving.END,
    )

# Configuration 10 - Pineapple - PIA Sieve

words = fruits.words.of_weight(4)

pineapple01 = fruits.Fruit("Pineapple_1")
pineapple01.add(fruits.preparation.INC)
pineapple01.add(*words)
pineapple01.branch().configure(mode="extended")
pineapple01.add(fruits.sieving.PIA([-1]))

pineapple01.fork()
pineapple01.add(*words)
pineapple01.branch().configure(mode="extended")
pineapple01.add(fruits.sieving.PIA([-1]))

pineapple02 = fruits.Fruit("Pineapple_2")
pineapple02.add(fruits.preparation.INC)
pineapple02.add(*words)
pineapple02.branch().configure(mode="extended")
pineapple02.add(fruits.sieving.PIA([0.5, -1]))

pineapple02.fork()
pineapple02.add(*words)
pineapple02.branch().configure(mode="extended")
pineapple02.add(fruits.sieving.PIA([0.5, -1]))

pineapple03 = fruits.Fruit("Pineapple_5")
pineapple03.add(fruits.preparation.INC)
pineapple03.add(*words)
pineapple03.branch().configure(mode="extended")
pineapple03.add(fruits.sieving.PIA([0.2, 0.4, 0.6, 0.8, -1]))

pineapple03.fork()
pineapple03.add(*words)
pineapple03.branch().configure(mode="extended")
pineapple03.add(fruits.sieving.PIA([0.2, 0.4, 0.6, 0.8, -1]))

pineapple04 = fruits.Fruit("Pineapple_10")
pineapple04.add(fruits.preparation.INC)
pineapple04.add(*words)
pineapple04.branch().configure(mode="extended")
pineapple04.add(fruits.sieving.PIA([i/10 for i in range(1, 11)]))

pineapple04.fork()
pineapple04.add(*words)
pineapple04.branch().configure(mode="extended")
pineapple04.add(fruits.sieving.PIA([i/10 for i in range(1, 11)]))

pineapple05 = fruits.Fruit("Pineapple_20")
pineapple05.add(fruits.preparation.INC)
pineapple05.add(*words)
pineapple05.branch().configure(mode="extended")
pineapple05.add(fruits.sieving.PIA([i/20 for i in range(1, 21)]))

pineapple05.fork()
pineapple05.add(*words)
pineapple05.branch().configure(mode="extended")
pineapple05.add(fruits.sieving.PIA([i/20 for i in range(1, 21)]))

# Configuration 11 - Lychee - LCS Sieve

words = fruits.words.of_weight(4)

lychee01 = fruits.Fruit("Lychee_1")
lychee01.add(fruits.preparation.INC)
lychee01.add(*words)
lychee01.branch().configure(mode="extended")
lychee01.add(fruits.sieving.LCS)

lychee01.fork()
lychee01.add(*words)
lychee01.branch().configure(mode="extended")
lychee01.add(fruits.sieving.LCS)

lychee02 = fruits.Fruit("Lychee_2")
lychee02.add(fruits.preparation.INC)
lychee02.add(*words)
lychee02.branch().configure(mode="extended")
lychee02.add(fruits.sieving.LCS([1, 0.5, -1], segments=True))

lychee02.fork()
lychee02.add(*words)
lychee02.branch().configure(mode="extended")
lychee02.add(fruits.sieving.LCS([1, 0.5, -1], segments=True))

lychee03 = fruits.Fruit("Lychee_5")
lychee03.add(fruits.preparation.INC)
lychee03.add(*words)
lychee03.branch().configure(mode="extended")
lychee03.add(fruits.sieving.LCS([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

lychee03.fork()
lychee03.add(*words)
lychee03.branch().configure(mode="extended")
lychee03.add(fruits.sieving.LCS([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

lychee04 = fruits.Fruit("Lychee_10")
lychee04.add(fruits.preparation.INC)
lychee04.add(*words)
lychee04.branch().configure(mode="extended")
lychee04.add(fruits.sieving.LCS([i/10 for i in range(0, 11)], segments=True))

lychee04.fork()
lychee04.add(*words)
lychee04.branch().configure(mode="extended")
lychee04.add(fruits.sieving.LCS([i/10 for i in range(0, 10)], segments=True))

lychee05 = fruits.Fruit("Lychee_20")
lychee05.add(fruits.preparation.INC)
lychee05.add(*words)
lychee05.branch().configure(mode="extended")
lychee05.add(fruits.sieving.LCS([i/20 for i in range(0, 21)], segments=True))

lychee05.fork()
lychee05.add(*words)
lychee05.branch().configure(mode="extended")
lychee05.add(fruits.sieving.LCS([i/20 for i in range(0, 21)], segments=True))

# Configuration 12 - Strawberry - Preparateur STD

words = fruits.words.of_weight(4)

strawberry = fruits.Fruit("Strawberry")
strawberry.add(fruits.preparation.STD)
strawberry.add(fruits.preparation.INC)
strawberry.add(*words)
strawberry.branch().configure(mode="extended")
strawberry.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

strawberry.fork()
strawberry.add(fruits.preparation.STD)
strawberry.add(*words)
strawberry.branch().configure(mode="extended")
strawberry.add(
    fruits.sieving.PPV,
    fruits.sieving.CPV,
    fruits.sieving.PIA,
    fruits.sieving.MAX,
    fruits.sieving.MIN,
    fruits.sieving.END,
)

basket = {
    "grape": (grape01, grape02, grape03),

    "apple": (apple01, apple02, apple03, apple04),

    "banana": (banana01, banana02, banana03, banana04, banana05, banana06),

    "plantain": (plantain01, plantain02, plantain03, plantain04, plantain05,
                 plantain06),

    "orange": (orange01, orange02, orange03, orange04),

    "tangerine": (tangerine01, tangerine02, tangerine03, tangerine04),

    "apricot": (apricot01, apricot02, apricot03, apricot04, apricot05, apricot06,
                apricot07, apricot08),

    "olive": (olive01, olive02, olive03),

    "elderberry": (elderberry01, elderberry02, elderberry03, elderberry04,
                   elderberry05),

    "dragonfruit": (dragonfruit01, dragonfruit02, dragonfruit03),

    "pineapple": (pineapple01, pineapple02, pineapple03, pineapple04,
                  pineapple05),

    "lychee": (lychee01, lychee02, lychee03, lychee04, lychee05),

    "strawberry": (strawberry, ),
}
