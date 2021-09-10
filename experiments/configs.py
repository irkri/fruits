import numpy as np

from context import fruits

np.random.seed(62)


# Configuration 01 - Apple - Number of words

words = fruits.core.generation.simplewords_by_weight(3)

apple01 = fruits.Fruit("Apple - Weight 3")
apple01.add(fruits.preparation.INC)
apple01.add(words)
apple01.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

apple01.fork()
apple01.add(words)
apple01.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

words = fruits.core.generation.simplewords_by_weight(4)

apple02 = fruits.Fruit("Apple - Weight 4")
apple02.add(fruits.preparation.INC)
apple02.add(words)
apple02.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

apple02.fork()
apple02.add(words)
apple02.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)

words = fruits.core.generation.simplewords_by_weight(4)

apple03 = fruits.Fruit("Apple - Weight 1-4")
apple03.add(fruits.preparation.INC)
apple03.add(words)
apple03.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple03.branch().calculator.mode = "extended"

apple03.fork()
apple03.add(words)
apple03.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple03.branch().calculator.mode = "extended"

words = fruits.core.generation.simplewords_by_weight(5)

apple04 = fruits.Fruit("Apple - Weight 1-5")
apple04.add(fruits.preparation.INC)
apple04.add(words)
apple04.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple04.branch().calculator.mode = "extended"

apple04.fork()
apple04.add(words)
apple04.add(fruits.sieving.PPV,
            fruits.sieving.PPVC,
            fruits.sieving.MAX,
            fruits.sieving.MIN,
            fruits.sieving.END)
apple04.branch().calculator.mode = "extended"


# Configuration 02 - Banana - Number of PPV/C Quantiles

words = fruits.core.generation.simplewords_by_weight(4)

banana01 = fruits.Fruit("Banana - PPV 0.5")
banana01.add(fruits.preparation.INC)
banana01.add(words)
banana01.branch().calculator.mode = "extended"
banana01.add(fruits.sieving.PPV(0.5))
banana01.add(fruits.sieving.PPVC(0.5))

banana01.fork()
banana01.add(words)
banana01.branch().calculator.mode = "extended"
banana01.add(fruits.sieving.PPV(0.5))
banana01.add(fruits.sieving.PPVC(0.5))

banana02 = fruits.Fruit("Banana - PPV 1-7 / 8")
banana02.add(fruits.preparation.INC)
banana02.add(words)
banana02.branch().calculator.mode = "extended"
banana02.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))
banana02.add(fruits.sieving.PPVC([i/8 for i in range(1, 8)]))

banana02.fork()
banana02.add(words)
banana02.branch().calculator.mode = "extended"
banana02.add(fruits.sieving.PPV([i/8 for i in range(1, 8)]))
banana02.add(fruits.sieving.PPVC([i/8 for i in range(1, 8)]))

banana03 = fruits.Fruit("Banana - PPV 1-9 / 10")
banana03.add(fruits.preparation.INC)
banana03.add(words)
banana03.branch().calculator.mode = "extended"
banana03.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))
banana03.add(fruits.sieving.PPVC([i/10 for i in range(1, 10)]))

banana03.fork()
banana03.add(words)
banana03.branch().calculator.mode = "extended"
banana03.add(fruits.sieving.PPV([i/10 for i in range(1, 10)]))
banana03.add(fruits.sieving.PPVC([i/10 for i in range(1, 10)]))

banana04 = fruits.Fruit("Banana - PPV 1-19 / 20")
banana04.add(fruits.preparation.INC)
banana04.add(words)
banana04.branch().calculator.mode = "extended"
banana04.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))
banana04.add(fruits.sieving.PPVC([i/20 for i in range(1, 20)]))

banana04.fork()
banana04.add(words)
banana04.branch().calculator.mode = "extended"
banana04.add(fruits.sieving.PPV([i/20 for i in range(1, 20)]))
banana04.add(fruits.sieving.PPVC([i/20 for i in range(1, 20)]))


# Configuration 03 - Orange - Number of cuts in MAX/MIN

words = fruits.core.generation.simplewords_by_weight(4)

orange01 = fruits.Fruit("Orange - MAX/MIN 2")
orange01.add(fruits.preparation.INC)
orange01.add(words)
orange01.branch().calculator.mode = "extended"
orange01.add(fruits.sieving.MAX([1, 0.5, -1], segments=True))
orange01.add(fruits.sieving.MIN([1, 0.5, -1], segments=True))

orange01.fork()
orange01.add(words)
orange01.branch().calculator.mode = "extended"
orange01.add(fruits.sieving.MAX([1, 0.5, -1], segments=True))
orange01.add(fruits.sieving.MIN([1, 0.5, -1], segments=True))

orange02 = fruits.Fruit("Orange - MAX/MIN 5")
orange02.add(fruits.preparation.INC)
orange02.add(words)
orange02.branch().calculator.mode = "extended"
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))
orange02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

orange02.fork()
orange02.add(words)
orange02.branch().calculator.mode = "extended"
orange02.add(fruits.sieving.MAX([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))
orange02.add(fruits.sieving.MIN([1, 0.2, 0.4, 0.6, 0.8, -1], segments=True))

orange03 = fruits.Fruit("Orange - MAX/MIN 10")
orange03.add(fruits.preparation.INC)
orange03.add(words)
orange03.branch().calculator.mode = "extended"
orange03.add(fruits.sieving.MAX([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))
orange03.add(fruits.sieving.MIN([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

orange03.fork()
orange03.add(words)
orange03.branch().calculator.mode = "extended"
orange03.add(fruits.sieving.MAX([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))
orange03.add(fruits.sieving.MIN([1]+[i/10 for i in range(1, 10)]+[-1],
                                segments=True))

orange04 = fruits.Fruit("Orange - MAX/MIN 20")
orange04.add(fruits.preparation.INC)
orange04.add(words)
orange04.branch().calculator.mode = "extended"
orange04.add(fruits.sieving.MAX([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))
orange04.add(fruits.sieving.MIN([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))

orange04.fork()
orange04.add(words)
orange04.branch().calculator.mode = "extended"
orange04.add(fruits.sieving.MAX([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))
orange04.add(fruits.sieving.MIN([1]+[i/20 for i in range(1, 20)]+[-1],
                                segments=True))


CONFIGS = [
    apple01, apple02, apple03, apple04,
    banana01, banana02, banana03, banana04,
    orange01, orange02, orange03, orange04,
]
