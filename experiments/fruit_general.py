import fruits

iss_r = fruits.ISS(
    fruits.words.of_weight(6, 2),
    mode=fruits.ISSMode.EXTENDED,
    semiring=fruits.iss.semiring.Reals(),
    weighting=fruits.iss.weighting.Indices(),
)

iss_a = fruits.ISS(fruits.words.alternate_sign([
        fruits.words.SimpleWord(48*"[1]"),
        fruits.words.SimpleWord(48*"[2]"),
        fruits.words.SimpleWord(24*"[1][2]"),
        fruits.words.SimpleWord(24*"[2][1]"),
    ]),
    mode=fruits.ISSMode.EXTENDED,
    semiring=fruits.semiring.Arctic(),
)

cos_words = (
    list(fruits.words.of_weight(1, 2))
    + list(fruits.words.of_weight(2, 2))
    + list(fruits.words.of_weight(3, 2))
    + list(fruits.words.of_weight(4, 2))
)


fruit = fruits.Fruit("General Fruit")

fruit.cut()
fruit.add(fruits.preparation.NEW(fruits.preparation.INC()))
fruit.add(fruits.preparation.STD)
fruit.add(iss_r)
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=0))
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=1))
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=2))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=0))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=1))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=2))
fruit.add(fruits.sieving.END)

fruit.cut()
fruit.add(fruits.preparation.NEW(fruits.preparation.INC()))
fruit.add(iss_a)
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=0))
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=1))
fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=2))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=0))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=1))
fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=2))
fruit.add(fruits.sieving.END)

for e in range(1, 3):
    fruit.cut()
    fruit.add(fruits.preparation.NEW(fruits.preparation.INC()))
    fruit.add(fruits.preparation.STD)
    fruit.add(fruits.CosWISS(
        freqs=[i/20 for i in range(1, 11, 2)],
        words=cos_words,
        exponent=e,
        total_weighting=True,
    ))
    fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=0))
    fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=1))
    fruit.add(fruits.sieving.NPI(q=(0.5, 1.0), inc=2))
    fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=0))
    fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=1))
    fruit.add(fruits.sieving.MPI(q=(0.5, 1.0), inc=2))
    fruit.add(fruits.sieving.END)

for slc in fruit:
    slc.fit_sample_size = 1.0

if __name__ == "__main__":
    print(fruit.nfeatures())
