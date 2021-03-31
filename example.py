from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS as sklearn_MDS
import time

df = pd.read_csv('khan_train.csv', index_col=0)
df_y = pd.read_csv('khan_train_y.csv')
X = df.to_numpy()
y = df_y['class'].to_numpy()

tick = time.perf_counter()
DELTA = minkowski_pairs(X, sqform = False)
mds = MDS(n_dims = 3, max_iter = 500, n_init = 3, verbosity = 1)
x = mds.fit(DELTA)
print('mdscuda time: ', time.perf_counter() - tick)
#print("mds r2: {}".format(mds.r2))

fig = px.scatter_3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], color=y, title='mdscuda')
fig.show()

tick = time.perf_counter()
embedding = sklearn_MDS(n_components = 3, max_iter=500, n_init = 3, verbose = 1)
X_transformed = embedding.fit_transform(X)
print('sklearn time: ', time.perf_counter() - tick)

fig = px.scatter_3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], color=y, title='sklearn')
fig.show()