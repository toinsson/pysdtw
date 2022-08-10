# Licensed under the MIT License, see LICENSE for details.
"""Torch implementation of the Soft-DTW algorithm. Include separate modules for
CPU and GPU computation and a dedicated module for distance functions.
"""
__version__ = '0.0.5'

import torch
import torch.nn.utils.rnn as rnn
from typing import Callable, Union, Tuple

from .sdtw_cuda import SoftDTWcuda
from .sdtw_cpu import SoftDTWcpu
from .distance import pairwise_l2_squared

T = torch.Tensor
PS = rnn.PackedSequence

class SoftDTW(torch.nn.Module):
    """Torch implementation of the Soft-DTW algorithm.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        dist_func: Callable = None,
        use_cuda: bool = True,
        bandwidth: int = None,
        ):
        """
        Args:
            gamma (float): Regularization parameter, lower is less smoothed (closer to true DTW).
            dist_func (func): Distance function used in pointwise computation, default to L2 squared.
            use_cuda (bool): Flag to use GPU, default to True.
            bandwidth (int): Sakoe-Chiba type bandwith parameter, default to 0.
        """
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func if dist_func is not None else pairwise_l2_squared
        self.use_cuda = use_cuda
        self.dtw_func = SoftDTWcuda.apply if use_cuda else SoftDTWcpu.apply

    def forward(self, X: Union[T, PS], Y: Union[T, PS]):
        """Compute the soft-DTW value between X and Y.

        Args:
            X (tensor or PackedSequence): input of size batch_size x seq_len_x x dims
            Y (tensor or PackedSequence): input of size batch_size x seq_len_y x dims

        Returns:
            The soft-DTW distance between X and Y of size batch_size.
        """
        X, Y, XY_lengths = _prepare_input(X, Y)
        XY_D = self.dist_func(X, Y)
        dtw = self.dtw_func(XY_D, XY_lengths, self.gamma, self.bandwidth)

        return dtw


def _prepare_input(x: Union[T, PS], y: Union[T, PS]) -> Tuple[T, T, T]:
    """Prepare the inputs. PackedSequences are unpacked. The lengths of
    individual sequences in x and y are returned as a staked array of shape
    (batchx2). Batch size and outer dimension of x and y must be the same.
    """
    x, x_len = _unpack_sequence(x)
    y, y_len = _unpack_sequence(y)
    xy_len = torch.stack([x_len, y_len]).T.to(x.device)

    bx, _, dx = x.shape
    by, _, dy = y.shape
    assert (bx == by) and (dx == dy)

    return x, y, xy_len

def _unpack_sequence(x: Union[T, PS]) -> Tuple[T, T]:
    """Return an unpacked sequence and lengths of subsequences.
    """
    if isinstance(x, rnn.PackedSequence):
        x, x_len = rnn.pad_packed_sequence(x, batch_first=True)
    else:
        u, v = x.shape[:2]
        x_len = torch.tensor([v]).expand(u)
    return x, x_len
