import numpy as np
import time
import plotly.express as px

from mdscuda.mds import MDS
from mdscuda.minkowski import minkowski_pairs

N_SAMPLES = 10000
N_FEATURES = 1000
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))

df = px.data.iris()
X_iris = df.iloc[:, :4].to_numpy()

def test_random_state():
    print()
    DELTA = minkowski_pairs(X_iris, sqform = False)
    mds = MDS(n_dims = 2, max_iter=10, n_init = 1, verbosity = 2, x_init= X_iris[:, :2])
    x = mds.fit(DELTA)
    #fig = px.scatter(x=x[:, 0], y=x[:, 1], color=df['species'], title='Iris mdscuda.MDS embedding')
    #fig.update_traces(marker=dict(size=9, opacity=.8))
    #fig.show()
    #print(x)
