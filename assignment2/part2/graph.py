import numpy as np

A = np.array([
[0,1,1,0,1],
[1,0,1,0,0],
[1,1,0,1,0],
[0,0,1,0,0],
[1,0,0,0,0]])

D = np.diag(np.diag(A @ A.T))

print(repr(D))
print(repr(A))
print(repr(A @ A.T))
print(repr(D - A))
