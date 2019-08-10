import numpy as np
from numba import cuda
from scipy.spatial.distance import squareform

USE_64 = False

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32
    
@cuda.jit("void(float{}[:, :], float{}, float{}[:], float{}[:])".format(bits, bits, bits, bits))
def distance_matrix_gpu(X, p, w, out):
    m = X.shape[0]
    n = X.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < j and j < m:
        for k in range(n):
            tmp = X[i, k] - X[j, k]
            d += w[k] * abs(tmp) ** p
        out[m * (m - 1) // 2 - (m - i) * (m - i - 1) // 2 + j - i - 1] = d ** (1/p)

def dist_matrix(X, p, w):
    rows = X.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))

    stream = cuda.stream()
    X = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    w2 = cuda.to_device(np.asarray(w, dtype = np_type), stream = stream)
    out2 = cuda.device_array(rows * (rows - 1) // 2, dtype = np_type)
    distance_matrix_gpu[grid_dim, block_dim](X, p, w2, out2)
    out = out2.copy_to_host(stream = stream)

    return out
    
def minkowski_distance(u, v, p = 2, w = 1):
    return np.linalg.norm(w**(1/p) * (u - v), ord = p)
    
def minkowski_pairs(X, p = 2, w = None, sqform = True):
    if w is None:
        w = np.ones(shape = (X.shape[1],))
    if sqform: 
        return squareform(dist_matrix(X, p, w))
    else:
        return dist_matrix(X, p, w)

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
    
def matmul(X, Y):
    assert X.shape[1] == Y.shape[0]
    rows = X.shape[0]
    cols = Y.shape[1]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(cols / block_dim[1] + 1))

    stream = cuda.stream()
    X2 = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    Y2 = cuda.to_device(np.asarray(Y, dtype = np_type), stream = stream)
    out2 = cuda.device_array((rows, cols), dtype = np_type)
    matmul_gpu[grid_dim, block_dim](X2, Y2, out2)
    out = out2.copy_to_host(stream = stream)

    return out

