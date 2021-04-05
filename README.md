# mdscuda
This is a CUDA implementation of Multidimensional Scaling (https://en.wikipedia.org/wiki/Multidimensional_scaling) using the SMACOF algorithm. Currently only metric MDS is supported. 

## Installation

pip install mdscuda

Latest version: 0.1.2

## Example

```Python
import numpy as np
from mdscuda import MDS, mds_fit, minkowski_pairs

N_SAMPLES = 1000
N_FEATURES = 100
X = np.random.normal(size = (N_SAMPLES, N_FEATURES))
DELTA = minkowski_pairs(X, sqform = False)  # this returns a matrix of pairwise distances in longform

# method 1: use an sklearn-style class

mds = MDS(n_dims = 3, verbosity = 2)  # defines sklearn-style class
x = mds.fit(DELTA)  # fits and returns embedding
print("mds r2: {}".format(mds.r2))  # prints R-squared value to assess quality of fit

# method 2: you can fit directly without using a class

x = mds_fit(DELTA, n_dims = 3, verbosity = 1)
```

## Documentation

class MDS methods:

mdscuda.MDS.init(self, n_dims = 2, max_iter = 300, n_init = 4, x_init = None, verbosity = 0)

* n_dims: int; number of dimensions in embedding space
* max_iter: int; maximum iterations of SMACOF algorithm to perform
* n_init: int; number of times to initialize SMACOF algorithm with random uniform(0, 100) initialization
* x_init: array or None; initial embedding. If not None, n_init is set to 1 and n_dims is set to x_init.shape[1]
* verbosity: int; if >= 1, print num iterations and final sigma values. if >= 2, print sigma value each iteration 
  (note: verbosity >= 2 slows performance by a factor of approximately 2)
    
mdscuda.MDS.fit(self, delta, sqform = False)

* delta: array; matrix of pairwise distances, longform by default, squareform if sqform == True
* sqform: bool; if True, delta is interpreted in squareformn, if False, delta is interpreted in longform

class MDS attributes: 

* mds.MDS.x: array or None; embedding
* mds.MDS.r2: float or None; R-squared value

mdscuda.mds_fit(delta, n_dims = 2, max_iter = 300, n_init = 4, x_init = None, verbosity = 0, sqform = False)

* delta: array; matrix of pairwise distances, longform by default, squareform if sqform == True
* n_dims: int; number of dimensions in embedding space
* max_iter: int; maximum iterations of SMACOF algorithm to perform
* n_init: int; number of times to initialize SMACOF algorithm with random uniform(0, 100) initialization
* x_init: array or None; initial embedding. If not None, n_init is set to 1 and n_dims is set to x_init.shape[1]
* verbosity: int; if >= 1, print num iterations and final sigma values. if >= 2, print sigma value each iteration
  (note: verbosity >= 2 slows performance by a factor of approximately 2)
* sqform: bool; if True, delta is interpreted in squareformn, if False, delta is interpreted in longform

mdscuda.minkowski_pairs(X, p = 2, w = None, sqform = True)

* X: array of shape (n_samples, n_features); matrix of samples
* p: float; p for minkowski distance
* w: array of shape (n_features, ) or None; weights for minkowski distance
* sqform: bool; if True, squareform matrix of pairwise distances is returned, otherwise, longform is returned

## Benchmarks

~~~text
Run on AMD Ryzen 5 2600 CPU and Nvidia RTX 2080 Ti GPU.
All times are in seconds.

Test1

Dataset: np.random.normal
Dataset shape: (10000, 1000)
Paramters: n_components = 3, max_iter = 50, n_init = 1

Results:
mdscuda final sigma: 11014284288.0
sklearn final sigma: 11022683577.672052

mdscuda time: 2.174905822990695
sklearn time: 125.06202016805764

Test2

Dataset: Iris
Dataset shape: (150, 4)
Parameters: n_components = 2, max_iter = 100, n_init = 100

Results:
mdscuda final sigma: 120.9668197631836
sklearn final sigma: 112.45789790236945

mdscuda time: 3.961419030005345
sklearn time: 3.922074425005121

Test3

Dataset: Digits
Dataset shape: (1797, 64)
Parameters: n_components = 3, max_iter = 600, n_init = 3

Results:
mdscuda final sigma: 199908928.0
sklearn final sigma: 199902115.6507256

mdscuda time:  2.1541129870165605
sklearn time:  90.4121356800024
~~~