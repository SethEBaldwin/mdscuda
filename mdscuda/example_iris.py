from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS as sklearn_MDS
import time

df = px.data.iris()
X = df.iloc[:, :4].to_numpy()

tick = time.perf_counter()
DELTA = minkowski_pairs(X, sqform = False)
mds = MDS(n_dims = 2, max_iter = 100, n_init = 100, verbosity = 1)
x = mds.fit(DELTA)
print("mds r2: {}".format(mds.r2))
print('mdscuda time: ', time.perf_counter() - tick)

fig = px.scatter(x=x[:, 0], y=x[:, 1], color=df['species'], title='mdscuda')
fig.show()

tick = time.perf_counter()
embedding = sklearn_MDS(n_components = 2, max_iter=100, n_init = 100, verbose = 1)
X_transformed = embedding.fit_transform(X)
print(embedding.stress_)
print('sklearn', time.perf_counter() - tick)

fig = px.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1], color=df['species'], title='sklearn')
fig.show()