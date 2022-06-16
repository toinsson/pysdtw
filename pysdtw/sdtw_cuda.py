from numba import cuda
import torch
from torch.autograd import Function
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, n_antidiag, R):
    """
    """
    inv_gamma = 1.0 / gamma

    b = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for a in range(n_antidiag):

        for p in range(n_passes):

            J = thread_id + p*MAX_THREADS_PER_BLOCK
            I = a - thread_id - p*MAX_THREADS_PER_BLOCK

            i = I + 1
            j = J + 1

            # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
            if (I + J == a) and (I < max_i and J < max_j) and (I > -1 and J > -1):

                # Don't compute if outside bandwidth
                if not (abs(i - j) > bandwidth > 0):
                    r0 = -R[b, i - 1, j - 1] * inv_gamma
                    r1 = -R[b, i - 1, j] * inv_gamma
                    r2 = -R[b, i, j - 1] * inv_gamma
                    rmax = max(max(r0, r1), r2)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads on this antidiagonal
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
MAX_THREADS_PER_BLOCK = 1024

class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        T = min(min(N, M), MAX_THREADS_PER_BLOCK)
        n_passes = min(N, M) // MAX_THREADS_PER_BLOCK + 1
        n_antidiag = M + N - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, T](
            cuda.as_cuda_array(D.detach()), gamma.item(), bandwidth.item(),
            N, M, n_passes, n_antidiag,
            cuda.as_cuda_array(R))

        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

