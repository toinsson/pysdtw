# Licensed under the MIT License, see LICENSE for details.

import torch
import numpy as np
from torch.autograd import Function
import numba


class SoftDTWcpu(Function):
    """CPU implementation of the Soft-DTW algorithm.
    """
    @staticmethod
    def forward(ctx, D, lengths, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()

        R = torch.Tensor(compute_softdtw(D_, lengths.numpy(), g_, b_)).to(dev).type(dtype)

        ctx.save_for_backward(D, R, lengths, gamma, bandwidth)

        Ms, Ns = lengths[:,0], lengths[:,1]
        res = R[:, Ms, Ns].diag()

        return res

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, lengths, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, lengths.numpy(), g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None


@numba.jit(nopython=True)
def sakoe_chiba_condition(i, j, M, N, bandwidth):
    """Approximate Sakoe-Chiba band for non-squared matrix.
    """
    i_sc, j_sc = i, j
    if N > M: i_sc = i * N / M
    if N < M: j_sc = j * M / N
    return (abs(i_sc - j_sc) > bandwidth > 0)


@numba.jit(nopython=True, parallel=True)
def compute_softdtw(D, lengths, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in numba.prange(B):
        N, M = lengths[b]
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                if sakoe_chiba_condition(i, j, N, M, bandwidth): continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@numba.jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, lengths, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_

    for k in numba.prange(B):

        Ni, Mi = lengths[k]
        E[k, Ni+1, Mi+1] = 1
        R[k, :, Mi+1] = -np.inf
        R[k, Ni+1, :] = -np.inf
        R[k, Ni+1, Mi+1] = R[k, Ni, Mi]

        for j in range(Mi, 0, -1):
            for i in range(Ni, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                if sakoe_chiba_condition(i, j, N, M, bandwidth): continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]
