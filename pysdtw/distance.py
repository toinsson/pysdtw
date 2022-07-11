# Licensed under the MIT License, see LICENSE for details.
"""Common distance functions.
"""
import torch
T = torch.Tensor

def pairwise_l2_squared(x: T, y: T) -> T:
    """Computes the pairwise distance matrix between x and y using the
    quadratic expansion. This limits the memory cost to the detriment of compute
    accuracy.
    """
    x_norm = (x**2).sum(-1).unsqueeze(-1)
    y_norm = (y**2).sum(-1).unsqueeze(-2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.mT)
    return torch.clamp(dist, 0.0, torch.inf)


def pairwise_l2_squared_exact(x: T, y: T) -> T:
    """Computes the pairwise distance matrix between x and y. This formula
    incurs a high memory cost.
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return torch.pow(x - y, 2).sum(3)
