from context import iss
import numpy as np

X = np.array([2,-4,3,7], dtype=np.float64)
Z = iss.get_increments(X, padding="left")

conc = iss.Concatination.from_str("[1][11][1][111][11111][1]")
ISS = iss.iterated_sums(Z, concatination=conc, verbose=True)

print(f"<{conc},ISS(Z)> = {ISS}")
print(f"ppv: {iss.ppv(ISS)}")