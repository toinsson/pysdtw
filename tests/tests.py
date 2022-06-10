# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("\tCPU:     ", avg_cpu)
    print("\tGPU:     ", avg_gpu)
    print("\tSpeedup: ", avg_cpu / avg_gpu)
    print()


# ----------------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     from timeit import default_timer as timer

#     torch.manual_seed(1234)

#     profile(128, 17, 15, 2, tol_backward=1e-6)
#     profile(512, 64, 64, 2, tol_backward=1e-4)
#     profile(512, 256, 256, 2, tol_backward=1e-3)



import torch
import unittest

class TestStringMethods(unittest.TestCase):

    def test_import(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(True)
        return True

    def test_import_cpu(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(False)
        return True

    def test_import_cuda(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(True)
        return True

    def test_equal(self):

        batch_size, seq_len_a, seq_len_b, dims = 10, 512, 1023, 15

        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        import pysdtw
        sdtw_cuda = pysdtw.SoftDTW(True)
        sdtw_cpu = pysdtw.SoftDTW(False)

        forward_cpu = sdtw_cpu(a_cpu, b_cpu)
        forward_gpu = sdtw_cuda(a_gpu, b_gpu)

        assert torch.allclose(forward_cpu, forward_gpu.cpu())

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()
