# Licensed under the MIT License, see LICENSE for details.

from numba import cuda
import torch
from torch.autograd import Function
import math


MAX_THREADS_PER_BLOCK = 1024


class SoftDTWcuda(Function):
    """CUDA implementation of the Soft-DTW algorithm.
    """
    @staticmethod
    def forward(ctx, D, lengths, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B, M, N = D.shape
        T = min(max(M, N), MAX_THREADS_PER_BLOCK)
        n_passes = max(M, N) // MAX_THREADS_PER_BLOCK + 1
        n_antidiag = M + N - 1

        R = torch.ones((B, M + 2, N + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        compute_softdtw_cuda[B, T](
            cuda.as_cuda_array(D.detach()), gamma.item(), bandwidth.item(),
            cuda.as_cuda_array(lengths), n_passes, n_antidiag,
            cuda.as_cuda_array(R)
            )

        ctx.save_for_backward(D, R.clone(), lengths, gamma, bandwidth)

        Ms, Ns = lengths[:,0], lengths[:,1]
        res = R[:, Ms, Ns].diag()

        return res

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, lengths, gamma, bandwidth = ctx.saved_tensors

        B, M, N = D.shape
        T = min(max(M, N), MAX_THREADS_PER_BLOCK)
        n_passes = max(M, N) // MAX_THREADS_PER_BLOCK + 1
        n_antidiag = M + N - 1

        D_ = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)
        D_[:, 1:M + 1, 1:N + 1] = D
        E = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)

        for Bi, (Mi, Ni) in enumerate(lengths):
            R[Bi, :, Ni+1] = -math.inf
            R[Bi, Mi+1, :] = -math.inf
            R[Bi, Mi+1, Ni+1] = R[Bi, Mi, Ni]
            E[Bi, Mi+1, Ni+1] = 1

        compute_softdtw_backward_cuda[B, T](
            cuda.as_cuda_array(D_), cuda.as_cuda_array(R), 1.0 / gamma.item(), bandwidth.item(),
            cuda.as_cuda_array(lengths), n_passes, n_antidiag,
            cuda.as_cuda_array(E)
            )

        E = E[:, 1:M + 1, 1:N + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None

@cuda.jit
def sakoe_chiba_condition(i, j, M, N, bandwidth):
    """Approximate Sakoe-Chiba band for non-squared matrix.
    """
    i_sc, j_sc = i, j
    if N > M: i_sc = i * N / M
    if N < M: j_sc = j * M / N
    return (abs(i_sc - j_sc) > bandwidth > 0)


@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, mn, n_passes, n_antidiag, R):
    inv_gamma = 1.0 / gamma

    Bi = cuda.blockIdx.x
    Mi, Ni = mn[Bi]
    thread_id = cuda.threadIdx.x

    for a in range(n_antidiag):
        for p in range(n_passes):

            I = a - thread_id - p*MAX_THREADS_PER_BLOCK
            J = thread_id + p*MAX_THREADS_PER_BLOCK

            if (I + J == a) and (I < Mi and J < Ni) and (I > -1):
                i, j = I + 1, J + 1

                if sakoe_chiba_condition(i, j, Mi, Ni, bandwidth): continue

                r0 = -R[Bi, i - 1, j - 1] * inv_gamma
                r1 = -R[Bi, i - 1, j] * inv_gamma
                r2 = -R[Bi, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[Bi, i, j] = D[Bi, i - 1, j - 1] + softmin

        cuda.syncthreads()


@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, mn, n_passes, n_antidiag, E):
    Bi = cuda.blockIdx.x
    Mi, Ni = mn[Bi]
    thread_id = cuda.threadIdx.x

    for a in range(n_antidiag):
        rev_a = n_antidiag - a - 1
        for p in range(n_passes):
            I = rev_a - thread_id - p*MAX_THREADS_PER_BLOCK
            J = thread_id + p*MAX_THREADS_PER_BLOCK

            if (I + J == rev_a) and (I < Mi and J < Ni) and (I > -1):
                i, j = I + 1, J + 1

                if math.isinf(R[Bi, i, j]):
                    R[Bi, i, j] = -math.inf

                if sakoe_chiba_condition(i, j, Mi, Ni, bandwidth): continue

                a = math.exp((R[Bi, i + 1, j] - R[Bi, i, j] - D[Bi, i + 1, j]) * inv_gamma)
                b = math.exp((R[Bi, i, j + 1] - R[Bi, i, j] - D[Bi, i, j + 1]) * inv_gamma)
                c = math.exp((R[Bi, i + 1, j + 1] - R[Bi, i, j] - D[Bi, i + 1, j + 1]) * inv_gamma)
                E[Bi, i, j] = E[Bi, i + 1, j] * a + E[Bi, i, j + 1] * b + E[Bi, i + 1, j + 1] * c

        cuda.syncthreads()
