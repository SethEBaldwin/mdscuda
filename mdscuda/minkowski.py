import numpy as np
from numba import cuda
from scipy.spatial.distance import squareform
from math import sqrt
import time

from mdscuda.utils import bits, np_type, idx, euclidean_pairs_gpu, euclidean_pairs_tiled_gpu, matmul_gpu, matmul_tiled_gpu

@cuda.jit("void(float{}[:, :], float{}, float{}[:], float{}[:])".format(bits, bits, bits, bits))
def distance_matrix_weighted_gpu(X, p, w, out):
    m = X.shape[0]
    n = X.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < j and j < m:
        for k in range(n):
            tmp = X[i, k] - X[j, k]
            d += w[k] * abs(tmp) ** p
        out[idx(i, j, m)] = d ** (1/p)

def dist_matrix_weighted(X, p, w):
    rows = X.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))

    stream = cuda.stream()
    X = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    w2 = cuda.to_device(np.asarray(w, dtype = np_type), stream = stream)
    out2 = cuda.device_array(rows * (rows - 1) // 2, dtype = np_type)
    distance_matrix_weighted_gpu[grid_dim, block_dim](X, p, w2, out2)
    out = out2.copy_to_host(stream = stream)

    return out

@cuda.jit("void(float{}[:, :], float{}, float{}[:])".format(bits, bits, bits))
def distance_matrix_gpu(X, p, out):
    m = X.shape[0]
    n = X.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < j and j < m:
        for k in range(n):
            tmp = X[i, k] - X[j, k]
            d += abs(tmp) ** p
        out[idx(i, j, m)] = d ** (1/p)

def dist_matrix(X, p):
    rows = X.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))

    stream = cuda.stream()
    X = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    out2 = cuda.device_array(rows * (rows - 1) // 2, dtype = np_type)
    if p == 2:
        tick = time.perf_counter()
        euclidean_pairs_gpu[grid_dim, block_dim](X, out2)
        print('euc pairs gpu', time.perf_counter() - tick)
    else:
        distance_matrix_gpu[grid_dim, block_dim](X, p, out2)
    out = out2.copy_to_host(stream = stream)

    return out

def dist_matrix_tiled(X, p):
    rows = X.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))

    stream = cuda.stream()
    X = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    out2 = cuda.device_array(rows * (rows - 1) // 2, dtype = np_type)
    if p == 2:
        tick = time.perf_counter()
        euclidean_pairs_tiled_gpu[grid_dim, block_dim](X, out2)
        print('euc pairs tiled gpu', time.perf_counter() - tick)
    else:
        distance_matrix_gpu[grid_dim, block_dim](X, p, out2)
    out = out2.copy_to_host(stream = stream)

    return out
    
def minkowski_pairs(X, p = 2, w = None, sqform = True):
    """return matrix where ij entry is minkowski distance between ith and jth row of X"""
    if w is None:
        if sqform: 
            return squareform(dist_matrix(X, p))
        else:
            return dist_matrix(X, p)
    if sqform: 
        return squareform(dist_matrix_weighted(X, p, w))
    else:
        return dist_matrix_weighted(X, p, w)

def minkowski_pairs_tiled(X, p = 2, w = None, sqform = True):
    """return matrix where ij entry is minkowski distance between ith and jth row of X"""
    if w is None:
        if sqform: 
            return squareform(dist_matrix_tiled(X, p))
        else:
            return dist_matrix_tiled(X, p)
    if sqform: 
        return squareform(dist_matrix_weighted(X, p, w))
    else:
        return dist_matrix_weighted(X, p, w)

# def minkowski_distance(u, v, p = 2, w = 1):
#     return np.linalg.norm(w**(1/p) * (u - v), ord = p)

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

def matmul_tiled(X, Y):
    assert X.shape[1] == Y.shape[0]
    rows = X.shape[0]
    cols = Y.shape[1]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(cols / block_dim[1] + 1))

    stream = cuda.stream()
    X2 = cuda.to_device(np.asarray(X, dtype = np_type), stream = stream)
    Y2 = cuda.to_device(np.asarray(Y, dtype = np_type), stream = stream)
    out2 = cuda.device_array((rows, cols), dtype = np_type)
    matmul_tiled_gpu[grid_dim, block_dim](X2, Y2, out2)
    out = out2.copy_to_host(stream = stream)

    return out