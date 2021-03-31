from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS as sklearn_MDS
import time
from sklearn.datasets import load_digits

N_SAMPLES = 10000
N_FEATURES = 1000
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))

tick = time.perf_counter()
DELTA = minkowski_pairs(X, sqform = False)
mds = MDS(n_dims = 3, max_iter = 50, n_init = 1, verbosity = 1)
x = mds.fit(DELTA)
print("mds r2: {}".format(mds.r2))
print('mdscuda time: ', time.perf_counter() - tick)

tick = time.perf_counter()
embedding = sklearn_MDS(n_components = 3, max_iter=50, n_init = 1, verbose = 2)
X_transformed = embedding.fit_transform(X)
print('sklearn time: ', time.perf_counter() - tick)
