import soft_dtw_cuda

import torch
import numpy as np


def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = soft_dtw_cuda.SoftDTW(False, gamma=1.0, normalize=False)
    # sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        # a_gpu = a_cpu.cuda()
        # b_gpu = b_cpu.cuda()

        # # GPU
        # t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = soft_dtw_cuda.timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        # assert torch.allclose(forward_cpu, forward_gpu.cpu())
        # assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            # times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    # avg_gpu = np.mean(times_gpu)
    print("\tCPU:     ", avg_cpu)
    # print("\tGPU:     ", avg_gpu)
    # print("\tSpeedup: ", avg_cpu / avg_gpu)
    print()

if __name__ == '__main__':
    profile(512, 256, 256, 15, 1e-3)