from context import iss
import numpy as np

X = np.array([2,-4,3,7])
Z = iss.get_increments(X, padding="left")

comp = "[1][11][1]"
ISS = iss.iterated_sums(Z, composition=comp, verbose=True)

print(f"<{comp},ISS(Z)> = {ISS}")
print(f"ppv: {iss.ppv(ISS)}")