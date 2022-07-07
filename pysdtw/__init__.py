# Licensed under the MIT License, see LICENSE for details.
"""Torch implementation of the Soft-DTW algorithm. Include separate modules for
CPU and GPU computation and a dedicated module for distance functions.
"""
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
        X, Y, XY_lengths = _check_input(X, Y)
        XY_D = self.dist_func(X, Y)
        dtw = self.dtw_func(XY_D, XY_lengths, self.gamma, self.bandwidth)

        return dtw


def _check_input(x: Union[T, PS], y: Union[T, PS]) -> Tuple[T, T, T]:
    """Checks the inputs. Batch size and outer dimension must be the same.
    """
    x, x_len, x_packed = _unpack_sequence(x)
    y, y_len, y_packed = _unpack_sequence(y)
    xy_len = torch.stack([x_len, y_len]).T if (x_packed or y_packed) else None

    bx, _, dx = x.shape
    by, _, dy = y.shape
    assert bx == by
    assert dx == dy

    return x, y, xy_len

def _unpack_sequence(x: Union[T, PS]) -> Tuple[T, T, bool]:
    if isinstance(x, rnn.PackedSequence):
        x, x_len = rnn.pad_packed_sequence(x, batch_first=True)
        packed = True
    else:
        u, v = x.shape[:2]
        x_len = torch.tensor([v]).expand(u)
        packed = False

    return x, x_len, packed