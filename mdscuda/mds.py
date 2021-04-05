import numpy as np
from numba import cuda
import time
from minkowski import minkowski_pairs
from scipy.spatial.distance import squareform
from scipy.stats.stats import pearsonr
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

@cuda.jit("void(float{}[:], float{}[:])".format(bits, bits))
def b_gpu(d, delta):
    i = cuda.grid(1)
    if i < d.shape[0]:
        if d[i] != 0: 
            d[i] = -delta[i] / d[i]

# TODO: tile?
@cuda.jit("void(float{}[:, :], float{}[:])".format(bits, bits))
def x_gpu(x, b):
    n = x.shape[0]
    m = x.shape[1]
    i, j = cuda.grid(2)
    if i < n and j < m:
        
        # k < i
        tmp = 0
        for k in range(i):
            tmp += b[idx(k, i, n)] * x[k, j]
        
        # k == i
        bii = 0
        for l in range(i):
            bii -= b[idx(l, i, n)]
        for l in range(i + 1, n):
            bii -= b[idx(i, l, n)]
        tmp += bii * x[i, j]
        
        # k > i
        for k in range(i + 1, n):
            tmp += b[idx(i, k, n)] * x[k, j]
        
        cuda.syncthreads()
        
        x[i, j] = tmp / n
        
@cuda.jit("void(float{}[:], float{}[:])".format(bits, bits))
def sigma_gpu(d, delta):
    i = cuda.grid(1)
    if i < d.shape[0]:
        tmp = d[i] - delta[i]
        d[i] = tmp * tmp
    
@cuda.jit("void(float{}[:], int32)".format(bits))
def sum_iter_gpu(d, s):
    i = cuda.grid(1)
    if i < s and i + s < d.shape[0]:
        d[i] += d[i + s]

# TODO: copying d[0] to host is extremely slow! any way around this?
def sigma(d, delta, blocks, tpb):
    tick = time.perf_counter()
    sigma_gpu[blocks, tpb](d, delta)
    print('sigma diff', time.perf_counter() - tick)
    tick = time.perf_counter()
    s = 1
    while s < d.shape[0]:
        s *= 2
    s = s // 2
    while s >= 1:
        sum_iter_gpu[int(s / tpb + 1), tpb](d, s)
        s = s // 2
    print('sigma sum', time.perf_counter() - tick)
    return d[0]
    
def smacof(x, delta, max_iter, verbosity):
    rows = x.shape[0]
    cols = x.shape[1]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))
    grid_dim_x = (int(rows / block_dim[0] + 1), int(cols / block_dim[1] + 1))

    tpb = 256
    grids = int(rows * (rows - 1) // 2 / tpb + 1)

    stream = cuda.stream()
    x2 = cuda.to_device(np.asarray(x, dtype = np_type), stream = stream)
    delta2 = cuda.to_device(np.asarray(delta, dtype = np_type), stream = stream)
    d2 = cuda.device_array(rows * (rows - 1) // 2, dtype = np_type)
    
    for iter in range(max_iter):
        
        if verbosity >= 2: #this overwrites d2
            euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
            tick = time.perf_counter()
            sig = sigma(d2, delta2, grids, tpb)
            print('sig', time.perf_counter() - tick)
            #todo: break condition.
            print("it: {}, sigma: {}".format(iter, sig))
        
        tick = time.perf_counter()
        euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
        print('euc', time.perf_counter() - tick)
        tick = time.perf_counter()
        b_gpu[grids, tpb](d2, delta2)
        print('b', time.perf_counter() - tick)
        tick = time.perf_counter()
        x_gpu[grid_dim_x, block_dim](x2, d2)
        print('bx', time.perf_counter() - tick)
    
    euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
    sig = sigma(d2, delta2, grids, tpb)
    
    if verbosity >= 2:
        print("it: {}, sigma: {}".format(iter + 1, sig))
    
    x = x2.copy_to_host(stream = stream)
    return (x, sig, iter)

# TODO: random state
# TODO: early stopping
def mds_fit(
    delta,              # matrix of pairwise distances of sample points 
                        #   in feature space, in longform by default
    n_dims = 2,         # number of dimensions of embedding space
    max_iter = 300,     # max number of iterations in smacof algorithm
    n_init = 4,         # number of times to run smacof
    x_init = None,      # initial embedding
                        #   if None, smacof is run on uniform(0, 100) initialization
                        #   if not None, n_init is set to 1
                        #   and n_dims is set to x_init.shape[1]
    verbosity = 0,      # if >= 1, print num iterations and final sigma values
                        #   if >= 2, print sigma value each iteration (slows performance)
    sqform = False      # if True, interpret delta as squareform
    ):

    tick = time.perf_counter()

    if sqform:
        assert delta.shape[0] == delta.shape[1]
        n_samples = delta.shape[0]
        delta = squareform(delta)
    
    else:
        n_samples = 1
        while n_samples * (n_samples - 1) // 2 != delta.shape[0]:
            if n_samples * (n_samples - 1) // 2 > delta.shape[0]:
                raise Exception('if sqform = False, delta must be given in longform')
            n_samples += 1

    if x_init is not None:
        assert x_init.shape[0] == n_samples
        n_init = 1
        n_dims = x_init.shape[1]
    
    for k in range(n_init):
        
        if x_init is None:
            x = np.random.uniform(0, 100, size = (n_samples, n_dims))
        else:
            x = x_init
        
        x, sig, iter = smacof(x, delta, max_iter, verbosity)
        
        if verbosity >= 1: 
            print("init {} lasted {} iterations. final sigma: {}"
                .format(k + 1, iter + 1, sig))
        
        if k == 0:
            best = (x, sig)
        elif sig < best[1]:
            best = (x, sig)
            
    if verbosity >= 1: 
        print("best sigma: {}".format(best[1]))
        print("mds total runtime: {} seconds".format(time.perf_counter() - tick))
    
    return best[0]

class MDS:  # sklearn style class
    
    def __init__(self, n_dims = 2, max_iter = 300, n_init = 4, x_init = None, verbosity = 0):
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.n_init = n_init
        self.x_init = x_init
        self.verbosity = verbosity
        self.x = None
        self.r2 = None
        self.delta = None
        self.sigma = None
    
    #TODO: record sigma
    def fit(self, delta, sqform = False):
        self.delta = delta
        self.x = mds_fit(
            self.delta, 
            n_dims = self.n_dims, 
            max_iter = self.max_iter, 
            n_init = self.n_init, 
            x_init = self.x_init, 
            verbosity = self.verbosity,
            sqform = sqform
        )
        if sqform:
            delta = squareform(delta)  # converts to longform for r2 calculation
        self.r2 = pearsonr(minkowski_pairs(self.x, sqform = False), delta)[0]**2
        return self.x
