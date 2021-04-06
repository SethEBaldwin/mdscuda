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

# unfortunately this isn't faster. 
# seems that the GPU gets warmed up after running once, so it appears faster if you run it after euclidean_pairs_gpu
# but reversing the order reverses the times...
@cuda.jit("void(float{}[:, :], float{}[:])".format(bits, bits))
def euclidean_pairs_tiled_gpu(x, d):

    TILE_SIZE = 16
    A = cuda.shared.array((TILE_SIZE, TILE_SIZE), np_type)
    B = cuda.shared.array((TILE_SIZE, TILE_SIZE), np_type)

    n = x.shape[0]
    m = x.shape[1]
    i, j = cuda.grid(2)
    
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y

    tmp = 0
    for s in range(int(m / TILE_SIZE + 1)):
        # copy to shared memory
        kA = tj + s*TILE_SIZE
        kB = ti + s*TILE_SIZE

        if kA < m and i < n:
            A[ti, tj] = x[i, kA]
        else:
            A[ti, tj] = 0
        if kB < m and j < n:
            B[ti, tj] = x[j, kB]
        else:
            B[ti, tj] = 0

        cuda.syncthreads()

        # compute partial sums
        if i < j:
            for k in range(TILE_SIZE):
                if k + s*TILE_SIZE < m and j < n:
                    diff = A[ti, k] - B[k, tj]
                else:
                    diff = 0
                tmp += diff * diff

        cuda.syncthreads()

    if i < j and j < n:
        d[idx(i, j, n)] = sqrt(tmp)

@cuda.jit("void(float{}[:, :], float{}[:, :], float{}[:, :])".format(bits, bits, bits))
def matmul_gpu(X, Y, out):
    n, p = out.shape[0], out.shape[1]
    m = X.shape[1]
    i, j = cuda.grid(2)
    c = 0
    if i < n and j < p:
        for k in range(m):
            c += X[i, k] * Y[k, j]
        out[i, j] = c

# unfortunately this is not faster
@cuda.jit("void(float{}[:, :], float{}[:, :], float{}[:, :])".format(bits, bits, bits))
def matmul_tiled_gpu(X, Y, out):
    TILE_SIZE = 16
    A = cuda.shared.array((TILE_SIZE, TILE_SIZE), np_type)
    B = cuda.shared.array((TILE_SIZE, TILE_SIZE), np_type)

    n = X.shape[0]
    m = X.shape[1]
    p = Y.shape[1]
    i, j = cuda.grid(2)

    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y

    tmp = 0
    for s in range(int(m / TILE_SIZE + 1)):
        # copy to shared memory
        kA = tj + s*TILE_SIZE
        kB = ti + s*TILE_SIZE

        if kA < m and i < n:
            A[ti, tj] = X[i, kA]
        else:
            A[ti, tj] = 0
        if kB < m and j < p:
            B[ti, tj] = Y[kB, j]
        else:
            B[ti, tj] = 0

        cuda.syncthreads()

        # compute partial sums
        for k in range(TILE_SIZE):
            tmp += A[ti, k] * B[k, tj]

        cuda.syncthreads()

    if i < n and j < p:
        out[i, j] = tmp
