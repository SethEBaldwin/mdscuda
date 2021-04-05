import numpy as np
#from mdscuda import MDS, mds_fit, minkowski_pairs
from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import time

N_SAMPLES = 10000
N_FEATURES = 1000
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))
tick = time.perf_counter()
DELTA = minkowski_pairs(X, sqform = False)  # this returns a matrix of pairwise distances in longform

# method 1: use an sklearn-style class

mds = MDS(n_dims = 3, max_iter=50, n_init=1, verbosity = 2)  # defines sklearn-style class
x = mds.fit(DELTA)  # fits and returns embedding
print('mds full runtime: {}'.format(time.perf_counter() - tick))
print("mds r2: {}".format(mds.r2))  # prints R-squared value to assess quality of fit

# method 2: you can fit directly without using a class

#x = mds_fit(DELTA, n_dims = 3, verbosity = 1)