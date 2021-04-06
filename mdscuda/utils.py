import numpy as np
from numba import cuda
import time
from scipy.spatial.distance import squareform
from math import sqrt

USE_64 = False

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32
    
# idx function explanation:
# Let A[:, :] be a symmetric matrix (squareform) of size n by n with zeros on the diagonal
# Let L[:] be the longform of A. Then for i < j, we have A[i, j] = L[idx(i, j, n)]
@cuda.jit('int32(int32, int32, int32)', device = True)
def idx(i, j, n):
    return n * (n - 1) // 2 - (n - i) * (n - i - 1) // 2 + j - i - 1

@cuda.jit("void(float{}[:, :], float{}[:])".format(bits, bits))
def euclidean_pairs_gpu(x, d):
    n = x.shape[0]
    m = x.shape[1]
    i, j = cuda.grid(2)
    if i < j and j < n:
        tmp = 0
        for k in range(m):
            diff = x[i, k] - x[j, k]
            tmp += diff * diff
        d[idx(i, j, n)] = sqrt(tmp)

# TODO: tile
@cuda.jit("void(float{}[:, :], float{}[:])".format(bits, bits))
def euclidean_pairs_tiled_gpu(x, d):
    n = x.shape[0]
    m = x.shape[1]
    i, j = cuda.grid(2)
    if i < j and j < n:
        tmp = 0
        for k in range(m):
            diff = x[i, k] - x[j, k]
            tmp += diff * diff
        d[idx(i, j, n)] = sqrt(tmp)

#A = cuda.shared.array((16, 16), "float{}".format(bits))
#B = cuda.shared.array((16, 16), "float{}".format(bits))