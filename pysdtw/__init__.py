import torch
import torch.cuda
from numba import cuda

from .sdtw_cuda import _SoftDTWCUDA
from .sdtw_cpu import _SoftDTW
from .distance import pairwise_l2_squared

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    """

    def __init__(self, gamma=1.0, normalize=False, bandwidth=None, dist_func=None, use_cuda=True):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.normalize = normalize
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func if dist_func is not None else pairwise_l2_squared
        self.use_cuda = use_cuda
        self.func_dtw = _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    def check_input(self, x, y):
        """Checks the inputs. Batch size must be the same.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy

    def forward(self, X, Y):
        """Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        self.check_input(X, Y)

        # this needs to be split for memory reasons
        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = self.func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)

        else:
            D_xy = self.dist_func(X, Y)
            return self.func_dtw(D_xy, self.gamma, self.bandwidth)
