# Licensed under the MIT License, see LICENSE for details.

import torch
import torch.cuda
from numba import cuda

from .sdtw_cuda import SoftDTWcuda
from .sdtw_cpu import SoftDTWcpu
from .distance import pairwise_l2_squared


class SoftDTW(torch.nn.Module):
    """Torch implementation of the Soft-DTW algorithm.
    """
    def __init__(self, gamma=1.0, dist_func=None, use_cuda=True, bandwidth=None, normalize=False):
        """
        Args:
            gamma (float): Regularization parameter, lower is less smoothed (closer to true DTW).
            dist_func (func): Distance function used in pointwise computation, default to L2 squared.
            use_cuda (bool): Flag to use GPU, default to True.
            normalize (bool): Flag to normalize input, default to True.
            bandwidth (int): Sakoe-Chiba bandwith parameter, default to 0.
        """
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.normalize = normalize
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func if dist_func is not None else pairwise_l2_squared
        self.use_cuda = use_cuda
        self.func_dtw = SoftDTWcuda.apply if use_cuda else SoftDTWcpu.apply

    def _check_input(self, x, y):
        """Checks the inputs. Batch size and outer dimension must be the same.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy

    def forward(self, X, Y):
        """Compute the soft-DTW value between X and Y.
        
        Args:
            X (tensor): input of size batch_size x seq_len_x x dims
            Y (tensor): input of size batch_size x seq_len_y x dims
        
        Returns: 
            The soft-DTW distance between X and Y of size batch_size.
        """
        self._check_input(X, Y)

        if not self.normalize:
            D_xy = self.dist_func(X, Y)
            return self.func_dtw(D_xy, self.gamma, self.bandwidth)

        # TODO: this should be checked for memory
        else:
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = self.func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
