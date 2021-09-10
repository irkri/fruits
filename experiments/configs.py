import numpy as np

from context import fruits

np.random.seed(62)

# Configuration 01 - Apple - Number of words

words = fruits.core.generation.simplewords_by_weight(3)

apple01 = fruits.Fruit("Apple_3")
apple01.add(fruits.preparation.INC)
apple01.add(words)
apple01.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

apple01.fork()
apple01.add(words)
apple01.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

words = fruits.core.generation.simplewords_by_weight(4)

apple02 = fruits.Fruit("Apple_4")
apple02.add(fruits.preparation.INC)
apple02.add(words)
apple02.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

apple02.fork()
apple02.add(words)
apple02.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

words = fruits.core.generation.simplewords_by_weight(4)

apple03 = fruits.Fruit("Apple_1_4")
apple03.add(fruits.preparation.INC)
apple03.add(words)
apple03.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple03.branch().calculator.mode = "extended"

apple03.fork()
apple03.add(words)
apple03.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple03.branch().calculator.mode = "extended"

words = fruits.core.generation.simplewords_by_weight(5)

apple04 = fruits.Fruit("Apple_1_5")
apple04.add(fruits.preparation.INC)
apple04.add(words)
apple04.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple04.branch().calculator.mode = "extended"

apple04.fork()
apple04.add(words)
apple04.add(fruits.sieving.PPV,
            fruits.sieving.PCC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple04.branch().calculator.mode = "extended"

# Configuration 02 - Banana - Number of PPV Quantiles

words = fruits.core.generation.simplewords_by_weight(4)

banana01 = fruits.Fruit("Banana_1")
banana01.add(fruits.preparation.INC)
banana01.add(words)
banana01.branch().calculator.mode = "extended"
banana01.add(fruits.sieving.PPV(0.5))

banana01.fork()
banana01.add(words)
banana01.branch().calculator.mode = "extended"
banana01.add(fruits.sieving.PPV(0.5))

banana02 = fruits.Fruit("Banana_3")
banana02.add(fruits.preparation.INC)
banana02.add(words)
banana02.branch().calculator.mode = "extended"
banana02.add(fruits.sieving.PPV([0.25, 0.5, 0.75]))

banana02.fork()
banana02.add(words)
banana02.branch().calculator.mode = "extended"
banana02.add(fruits.sieving.PPV([0.25, 0.5, 0.75]))

banana03 = fruits.Fruit("Banana_7")
banana03.add(fruits.preparation.INC)
banana03.add(words)
banana03.branch().calculator.mode = "extended"
banana03.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))

banana03.fork()
banana03.add(words)
banana03.branch().calculator.mode = "extended"
banana03.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))

banana04 = fruits.Fruit("Banana_9")
banana04.add(fruits.preparation.INC)
banana04.add(words)
banana04.branch().calculator.mode = "extended"
banana04.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))

banana04.fork()
banana04.add(words)
banana04.branch().calculator.mode = "extended"
banana04.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))

banana05 = fruits.Fruit("Banana_19")
banana05.add(fruits.preparation.INC)
banana05.add(words)
banana05.branch().calculator.mode = "extended"
banana05.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))

banana05.fork()
banana05.add(words)
banana05.branch().calculator.mode = "extended"
banana05.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))

# Configuration 03 - Plantain - Number of PCC Quantiles

words = fruits.core.generation.simplewords_by_weight(4)

plantain01 = fruits.Fruit("Plantain_1")
plantain01.add(fruits.preparation.INC)
plantain01.add(words)
plantain01.branch().calculator.mode = "extended"
plantain01.add(fruits.sieving.PCC(0.5))

plantain01.fork()
plantain01.add(words)
plantain01.branch().calculator.mode = "extended"
plantain01.add(fruits.sieving.PCC(0.5))

plantain02 = fruits.Fruit("Plantain_3")
plantain02.add(fruits.preparation.INC)
plantain02.add(words)
plantain02.branch().calculator.mode = "extended"
plantain02.add(fruits.sieving.PCC([0.25, 0.5, 0.75]))

plantain02.fork()
plantain02.add(words)
plantain02.branch().calculator.mode = "extended"
plantain02.add(fruits.sieving.PCC([0.25, 0.5, 0.75]))

plantain03 = fruits.Fruit("Plantain_7")
plantain03.add(fruits.preparation.INC)
plantain03.add(words)
plantain03.branch().calculator.mode = "extended"
plantain03.add(fruits.sieving.PCC([i/8 for i in range(1, 8)]))

plantain03.fork()
plantain03.add(words)
plantain03.branch().calculator.mode = "extended"
plantain03.add(fruits.sieving.PCC([i/8 for i in range(1, 8)]))

plantain04 = fruits.Fruit("Plantain_9")
plantain04.add(fruits.preparation.INC)
plantain04.add(words)
plantain04.branch().calculator.mode = "extended"
plantain04.add(fruits.sieving.PCC([i/10 for i in range(1, 10)]))

plantain04.fork()
plantain04.add(words)
plantain04.branch().calculator.mode = "extended"
plantain04.add(fruits.sieving.PCC([i/10 for i in range(1, 10)]))

plantain05 = fruits.Fruit("Plantain_19")
plantain05.add(fruits.preparation.INC)
plantain05.add(words)
plantain05.branch().calculator.mode = "extended"
plantain05.add(fruits.sieving.PCC([i/20 for i in range(1, 20)]))

plantain05.fork()
plantain05.add(words)
plantain05.branch().calculator.mode = "extended"
plantain05.add(fruits.sieving.PCC([i/20 for i in range(1, 20)]))

# Configuration 04 - Orange - Number of cuts in MAX

words = fruits.core.generation.simplewords_by_weight(4)

orange01 = fruits.Fruit("Orange_2")
orange01.add(fruits.preparation.INC)
orange01.add(words)
orange01.branch().calculator.mode = "extended"
orange01.add(fruits.sieving.MAX([1, 0.5, -1], segments=True))

orange01.fork()
orange01.add(words)
orange01.branch().calculator.mode = "extended"
orange01.add(fruits.sieving.MAX([1, 0.5, -1], segments=True))

orange02 = fruits.Fruit("Orange_5")
orange02.add(fruits.preparation.INC)
orange02.add(words)
orange02.branch().calculator.mode = "extended"
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

orange02.fork()
orange02.add(words)
orange02.branch().calculator.mode = "extended"
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

orange03 = fruits.Fruit("Orange_10")
orange03.add(fruits.preparation.INC)
orange03.add(words)
orange03.branch().calculator.mode = "extended"
orange03.add(fruits.sieving.MAX([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

orange03.fork()
orange03.add(words)
orange03.branch().calculator.mode = "extended"
orange03.add(fruits.sieving.MAX([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

orange04 = fruits.Fruit("Orange_20")
orange04.add(fruits.preparation.INC)
orange04.add(words)
orange04.branch().calculator.mode = "extended"
orange04.add(fruits.sieving.MAX([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))

orange04.fork()
orange04.add(words)
orange04.branch().calculator.mode = "extended"
orange04.add(fruits.sieving.MAX([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))

# Configuration 05 - Tangerine - Number of cuts in MIN

words = fruits.core.generation.simplewords_by_weight(4)

tangerine01 = fruits.Fruit("Tangerine_2")
tangerine01.add(fruits.preparation.INC)
tangerine01.add(words)
tangerine01.branch().calculator.mode = "extended"
tangerine01.add(fruits.sieving.MIN([1, 0.5, -1], segments=True))

tangerine01.fork()
tangerine01.add(words)
tangerine01.branch().calculator.mode = "extended"
tangerine01.add(fruits.sieving.MIN([1, 0.5, -1], segments=True))

tangerine02 = fruits.Fruit("Tangerine_5")
tangerine02.add(fruits.preparation.INC)
tangerine02.add(words)
tangerine02.branch().calculator.mode = "extended"
tangerine02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

tangerine02.fork()
tangerine02.add(words)
tangerine02.branch().calculator.mode = "extended"
tangerine02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

tangerine03 = fruits.Fruit("Tangerine_10")
tangerine03.add(fruits.preparation.INC)
tangerine03.add(words)
tangerine03.branch().calculator.mode = "extended"
tangerine03.add(fruits.sieving.MIN([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

tangerine03.fork()
tangerine03.add(words)
tangerine03.branch().calculator.mode = "extended"
tangerine03.add(fruits.sieving.MIN([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

tangerine04 = fruits.Fruit("Tangerine_20")
tangerine04.add(fruits.preparation.INC)
tangerine04.add(words)
tangerine04.branch().calculator.mode = "extended"
tangerine04.add(fruits.sieving.MIN([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))

tangerine04.fork()
tangerine04.add(words)
tangerine04.branch().calculator.mode = "extended"
tangerine04.add(fruits.sieving.MIN([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))


CONFIGS = [
    apple01, apple02, apple03, apple04,
    banana01, banana02, banana03, banana04, banana05,
    plantain01, plantain02, plantain03, plantain04, plantain05,
    orange01, orange02, orange03, orange04,
    tangerine01, tangerine02, tangerine03, tangerine04,
]
