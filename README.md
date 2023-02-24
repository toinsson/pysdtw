# pysdtw - [![Downloads](https://pepy.tech/badge/pysdtw)](https://pepy.tech/project/pysdtw)

Torch implementation of the [Soft-DTW algorithm](https://github.com/mblondel/soft-dtw) with support for both cpu and CUDA hardware.

This repository started as a fork from this [project](https://github.com/Maghoumi/pytorch-softdtw-cuda), but now exists as a stand-alone to include several improvements:
- availability on [pypi](https://pypi.org/project/pysdtw/)
- code organisation as a package
- improved API with type declaration
- support for time series of arbitrary lengths on CUDA
- support for packed sequences
- fixes for Sakoe-Ichiba bands



# Installation

This package is available on [pypi](https://pypi.org/project/pysdtw/) and depends on `pytorch` and `numba`.

Install with:

`pip install pysdtw`

# Usage

Below is a small snippet showcasing the computation of the DTW between two batched tensors which also yields the gradient of the DTW with regards to one of the inputs:
```python
import torch
import pysdtw

device=torch.device('cuda')

# the input data includes a batch dimension
X = torch.rand((10, 5, 7), device=device, requires_grad=True)
Y = torch.rand((10, 9, 7), device=device)

# optionally choose a pairwise distance function
fun = pysdtw.distance.pairwise_l2_squared

# create the SoftDTW distance function
sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=True)

# soft-DTW discrepancy, approaches DTW as gamma -> 0
res = sdtw(X, Y)

# define a loss, which gradient can be backpropagated
loss = res.sum()
loss.backward()

# X.grad now contains the gradient with respect to the loss
```

You can also have a look at the code in the tests directory. Different test suites ensure that [pysdtw](https://github.com/toinsson/pysdtw/) behaves similarly to [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Maghoumi and [soft-dtw](https://github.com/mblondel/soft-dtw) by Blondel. The tests also include some comparative performance measurements. Run the tests with `python -m unittests` from the root.

# Acknowledgements

Supported by the ELEMENT project (ANR-18-CE33-0002) and the ARCOL project (ANR-19-CE33-0001) from the French National Research Agency
