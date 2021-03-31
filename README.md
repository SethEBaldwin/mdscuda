# MDScuda
This is a cuda implementation of Multidimensional Scaling (https://en.wikipedia.org/wiki/Multidimensional_scaling) using the SMACOF algorithm. Currently only metric MDS is supported. 

## Example:

```Python
import numpy as np
from mds import MDS, mds_fit
from minkowski import minkowski_pairs

N_SAMPLES = 1000
N_FEATURES = 100
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))
DELTA = minkowski_pairs(X, sqform = False) #this returns a matrix of pairwise distances in longform

#method 1: use an sklearn-style class

mds = MDS(n_dims = 3, verbosity = 2) #defines sklearn-style class
x = mds.fit(DELTA) #fits and returns embedding
print("mds r2: {}".format(mds.r2)) #prints R-squared value to assess quality of fit

#method 2: you can fit directly without using a class

x = mds_fit(DELTA, n_dims = 3, verbosity = 1)
```

## Documentation for mds.py:

class MDS methods:

mds.MDS.init(self, n_dims = 2, max_iter = 300, n_init = 4, x_init = None, verbosity = 0)

* n_dims: int; number of dimensions in embedding space
* max_iter: int; maximum iterations of SMACOF algorithm to perform
* n_init: int; number of times to initialize SMACOF algorithm with random uniform(0, 100) initialization
* x_init: array or None; initial embedding. If not None, n_init is set to 1 and n_dims is set to x_init.shape[1]
* verbosity: int; if >= 1, print num iterations and final sigma values. if >= 2, print sigma value each iteration 
  (note: verbosity >= 2 slows performance by a factor of approximately 2)
    
mds.MDS.fit(self, delta, sqform = False)

* delta: array; matrix of pairwise distances, longform by default, squareform if sqform == True
* sqform: bool; if True, delta is interpreted in squareformn, if False, delta is interpreted in longform

class MDS attributes: 

* mds.MDS.x: array or None; embedding
* mds.MDS.r2: float or None; R-squared value

Other functions from mds.py:

mds.mds_fit(delta, n_dims = 2, max_iter = 300, n_init = 4, x_init = None, verbosity = 0, sqform = False)

* delta: array; matrix of pairwise distances, longform by default, squareform if sqform == True
* n_dims: int; number of dimensions in embedding space
* max_iter: int; maximum iterations of SMACOF algorithm to perform
* n_init: int; number of times to initialize SMACOF algorithm with random uniform(0, 100) initialization
* x_init: array or None; initial embedding. If not None, n_init is set to 1 and n_dims is set to x_init.shape[1]
* verbosity: int; if >= 1, print num iterations and final sigma values. if >= 2, print sigma value each iteration
  (note: verbosity >= 2 slows performance by a factor of approximately 2)
* sqform: bool; if True, delta is interpreted in squareformn, if False, delta is interpreted in longform

                    
## Documentation for minkowski.py:

minkowski.minkowski_pairs(X, p = 2, w = None, sqform = True)

* X: array of shape (n_samples, n_features); matrix of samples
* p: float; p for minkowski distance
* w: array of shape (n_features, ) or None; weights for minkowski distance
* sqform: bool; if True, squareform matrix of pairwise distances is returned, otherwise, longform is returned
