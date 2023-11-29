import fruits

fruit = fruits.Fruit("Time Warping Invariant Fruit")

fruit.cut()
fruit.add(fruits.preparation.INC)
fruit.add(fruits.ISS(
    fruits.words.of_weight(9, 1),
    mode=fruits.ISSMode.EXTENDED,
    semiring=fruits.semiring.Reals(),
    weighting=fruits.iss.weighting.L1(),
))
fruit.add(
    fruits.sieving.NPI,
    fruits.sieving.MPI,
    fruits.sieving.END,
)

fruit.cut()
fruit.add(fruits.ISS(
    fruits.words.alternate_sign([fruits.words.SimpleWord(48*"[1]")]),
    mode=fruits.ISSMode.EXTENDED,
    semiring=fruits.semiring.Arctic(),
))
fruit.add(
    fruits.sieving.NPI,
    fruits.sieving.END,
)

for slc in fruit:
    slc.fit_sample_size = 1.0

if __name__ == "__main__":
    print(fruit.nfeatures())
