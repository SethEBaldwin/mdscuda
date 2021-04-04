from mds import MDS, mds_fit
from minkowski import minkowski_pairs
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS as sklearn_MDS
import time
from sklearn.datasets import load_digits
import imageio
from math import sin, cos
import io 
from PIL import Image

def plotly_fig_to_array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

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
xs = [1.768*cos(theta + .785) for theta in np.arange(0, 6.283, 0.0523)]
ys = [1.768*sin(theta + .785) for theta in np.arange(0, 6.283, 0.0523)]
eyes = [{'x': x, 'y': y, 'z': 1.25} for x, y in zip(xs, ys)]
cameras = [{'eye': eye} for eye in eyes]
pixels_list = []
for camera in cameras:
    fig.update_traces()
    fig.update_layout(scene_camera=camera)
    fig.update_scenes(
        xaxis={'showticklabels': False, 'title': ""}, 
        yaxis={'showticklabels': False, 'title': ""}, 
        zaxis={'showticklabels': False, 'title': ""}
    )
    pixels = plotly_fig_to_array(fig)
    pixels = pixels[80:-50, 165:-165, :]
    pixels_list.append(pixels)
imageio.mimsave('digits-mdscuda.gif', pixels_list, duration = 0.075)
#fig.show()
#fig.write_html("digits-mdscuda.html")

# tick = time.perf_counter()
# embedding = sklearn_MDS(n_components = 3, max_iter=600, n_init = 3, verbose = 1)
# X_transformed = embedding.fit_transform(X)
# print('sklearn time: ', time.perf_counter() - tick)

# fig = px.scatter_3d(x=X_transformed[:, 0], y=X_transformed[:, 1], z=X_transformed[:, 2], color=y, title='Digits sklearn.manifold.MDS embedding')
# fig.update_traces(marker=dict(size=6))
# fig.show()
# fig.write_html("digits-sklearn.html")
