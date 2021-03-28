from context import iss
import numpy as np

X = np.array([1,3,2,8])
Z = iss.get_increments(X)

print(f"X: {str(X)}")
print(f"Z: {str(Z)}")

comp = "[1]"
ISS = iss.iterated_sums(Z, composition=comp)

print(f"<{comp},ISS(Z)> = {ISS}")
print(f"ppv: {iss.ppv(ISS)}")