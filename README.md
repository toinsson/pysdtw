# pysdtw

Torch implementation of the [Soft-DTW algorithm](https://github.com/mblondel/soft-dtw), supports both cpu and CUDA hardware.

Note: This repository started as a fork from this [project](https://github.com/Maghoumi/pytorch-softdtw-cuda).

# Installation

This package is available on [pypi](https://pypi.org/project/pysdtw/) and depends on `pytorch` and `numba`.

Install with:

`pip install pysdtw`

# Usage

```python
import pysdtw

# the input data includes a batch dimension
X = torch.rand((10, 5, 7), requires_grad=True)
Y = torch.rand((10, 9, 7))

# optionally choose a pairwise distance function
fun = pysdtw.distance.pairwise_l2_squared

# create the SoftDTW distance function
sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=False)

# soft-DTW discrepancy, approaches DTW as gamma -> 0
res = sdtw(X, Y)

# define a loss, which gradient can be backpropagated
loss = res.sum()
loss.backward()

# X.grad now contains the gradient with respect to the loss
```
