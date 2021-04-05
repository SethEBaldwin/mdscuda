from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS as sklearn_MDS
import time
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
df = pd.DataFrame(data=X, columns=range(64), index=range(1797))
df['digit'] = y
df = df.sort_values('digit')
X = df.drop('digit', axis=1).to_numpy()
y = df['digit'].astype(str).to_numpy()
#print(X.shape)
#print(y.shape)
#print(X)
#print(y)

tick = time.perf_counter()
DELTA = minkowski_pairs(X, sqform = False)
mds = MDS(n_dims = 3, max_iter = 600, n_init = 3, verbosity = 1)
x = mds.fit(DELTA)
print('mdscuda time: ', time.perf_counter() - tick)

fig = px.scatter_3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], color=y, title='Digits mdscuda.MDS embedding')
fig.update_traces(marker=dict(size=6))
fig.show()
#fig.write_html("digits-mdscuda.html")

tick = time.perf_counter()
embedding = sklearn_MDS(n_components = 3, max_iter=600, n_init = 3, verbose = 1)
X_transformed = embedding.fit_transform(X)
print('sklearn time: ', time.perf_counter() - tick)

fig = px.scatter_3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], color=y, title='Digits sklearn.manifold.MDS embedding')
fig.update_traces(marker=dict(size=6))
fig.show()
#fig.write_html("digits-sklearn.html")
