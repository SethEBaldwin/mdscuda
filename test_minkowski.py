import numpy as np
import time

from mdscuda.minkowski import minkowski_pairs, minkowski_pairs_tiled, matmul, matmul_tiled

N_SAMPLES = 10000
N_FEATURES = 1000
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))

A = np.random.normal(size = (1000, 500))
B = np.random.normal(size = (500, 1000))
#A = np.full(shape = (1000, 500), fill_value=3)
#B = np.full(shape = (500, 1000), fill_value=4)

def test_euclidean_tiled():
    print()
    tick = time.perf_counter()
    DELTA = minkowski_pairs(X, sqform = False)
    print('euc pairs', time.perf_counter() - tick)
    #print(DELTA)

    tick = time.perf_counter()
    DELTA_tiled = minkowski_pairs_tiled(X, sqform = False)
    print('euc pairs tiled', time.perf_counter() - tick)
    #print(DELTA_tiled)

    epsilon = 1e-5
    #print((np.abs(DELTA - DELTA_tiled)).max())
    within_epsilon = (np.abs(DELTA - DELTA_tiled) < epsilon).all()
    print('within epsilon', within_epsilon)
    assert within_epsilon

def test_matmul_tiled():
    print()
    tick = time.perf_counter()
    AB = matmul(A, B)
    print('matmul', time.perf_counter() - tick)
    #print(AB)

    tick = time.perf_counter()
    AB_tiled = matmul_tiled(A, B)
    print('matmul tiled', time.perf_counter() - tick)
    #print(AB_tiled)

    epsilon = 1e-8
    #print((np.abs(AB - AB_tiled)).max())
    within_epsilon = (np.abs(AB - AB_tiled) < epsilon).all()
    print('within epsilon', within_epsilon)
    assert within_epsilon