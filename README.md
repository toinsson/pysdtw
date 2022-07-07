# pysdtw

Torch implementation of the [Soft-DTW algorithm](https://github.com/mblondel/soft-dtw), supports both cpu and gpu hardware.

This repository started as a fork from this [project](https://github.com/Maghoumi/pytorch-softdtw-cuda).
The main source is included in the tests repository.

# Installation

This package is available on [pypi](https://pypi.org/project/pysdtw/) and depends on `pytorch` and `numba`. 
Install with:

`pip install pysdtw`

# Usage

```
import pysdtw

# the input data includes a batch dimension
X = torch.rand((10, 5, 7))
Y = torch.rand((10, 9, 7))

# optionally choose a pairwise distance function
fun = pysdtw.distance.pairwise_l2_squared

# create the SoftDTW distance function
sdtw = SoftDTW(D, gamma=1.0, dist_func=fun, use_cuda=False)

# soft-DTW discrepancy, approaches DTW as gamma -> 0
res = sdtw(X, Y)

# gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
E = sdtw.grad()
```
