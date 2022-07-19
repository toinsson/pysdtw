import unittest
import torch
import pysdtw

def timed_run(n_iters, a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer
    
    # warmup
    forward = sdtw(a, b)
    grad_outputs = torch.ones_like(forward)
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    
    t = 0

    for i in range(n_iters):
        start = timer()
        forward = sdtw(a, b)
        end = timer()
        t += end - start

        grad_outputs = torch.ones_like(forward)

        # Backward
        start = timer()
        grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
        end = timer()

        # Total time
        t += end - start

    return t, forward, grads

def timed_run_packed(n_iters, a_packed, b_packed, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer
    
    # warmup
    forward = sdtw(a_packed, b_packed)
    # grad_outputs = torch.ones_like(forward)
    # grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    
    t = 0

    for i in range(n_iters):
        start = timer()
        forward = sdtw(a_packed, b_packed)
        end = timer()
        t += end - start

        # grad_outputs = torch.ones_like(forward)

        # Backward
        # start = timer()
        # grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
        # end = timer()

        # Total time
        # t += end - start

    return t

class TestPerf(unittest.TestCase):
    def setUp(self):
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
    
    def test_cpu_gpu(self):
        
        batch_size, seq_len_a, seq_len_b, dims = 100, 500, 300, 12

        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)

        n_iters = 10
        t_gpu, forward_gpu, backward_gpu = timed_run(n_iters, a_gpu, b_gpu, sdtw_cuda)
        t_cpu, forward_cpu, backward_cpu = timed_run(n_iters, a_cpu, b_cpu, sdtw_cpu)

        print("GPU:{:.2f} CPU:{:.2f} RATIO:{:.2f}".format(t_gpu, t_cpu, t_gpu/t_cpu))


    def test_packed(self):
        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        import numba
        import torch.nn.utils.rnn as rnn

        batch_size = 100
        dims = 15

        # batch_size, seq_len_a, seq_len_b, dims = 100, 1500, 1300, 12

        from timeit import default_timer as timer
        t_itr = 0
        n_iters = 100
        
        for i in numba.prange(n_iters):
            
            len_a = torch.randint(10, 100, (batch_size,))
            len_b = torch.randint(10, 100, (batch_size,))
            a = [torch.rand((l, dims), requires_grad=True) for l in len_a]
            b = [torch.rand((l, dims)) for l in len_b]
            
            start = timer()
            res0 = torch.cat([sdtw_cuda(ai.unsqueeze(0).cuda(), 
                                        bi.unsqueeze(0).cuda()) for (ai, bi) in zip(a, b)])
            t_itr += timer() - start
        
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b_packed = rnn.pack_sequence(b, enforce_sorted=False)
        t_pck = timed_run_packed(n_iters, a_packed.cuda(), b_packed.cuda(), sdtw_cuda)
        # t_cpu, forward_cpu, backward_cpu = timed_run(n_iters, a_cpu, b_cpu, sdtw_cpu)

        print("PACKED:{:.2f} ITER:{:.2f} RATIO:{:.2f}".format(t_pck, t_itr, t_pck/t_itr))    

    def test_legacy_gpu(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda

        batch_size, seq_len_a, seq_len_b, dims = 10, 50, 30, 12

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))
        Bc = torch.rand((batch_size, seq_len_b, dims))
        
        sdtw_leg = soft_dtw_cuda.SoftDTW(True)
        sdtw_gpu = pysdtw.SoftDTW(use_cuda=True)

        n_iters = 100
        Acuda, Accuda, Bcuda, Bccuda = A.cuda(), Ac.cuda(), B.cuda(), Bc.cuda()

        t_leg, _, _ = timed_run(n_iters, Acuda, Bcuda, sdtw_leg)
        t_gpu, _, _ = timed_run(n_iters, Accuda, Bccuda, sdtw_gpu)

        
        print("PYSDTW:{:.2f} MAGHOUMI:{:.2f} RATIO:{:.2f}".format(t_gpu, t_leg, t_gpu/t_leg))
        
    def test_cython(self):
        
        import numba
        
        from sdtw import SoftDTW
        from sdtw.distance import SquaredEuclidean
        import numpy as np
        
        def fun(X, Y):
            D = SquaredEuclidean(X, Y)
            sdtw = SoftDTW(D, gamma=1.0)
            # soft-DTW discrepancy, approaches DTW as gamma -> 0
            value = sdtw.compute()
            # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
            E = sdtw.grad()
            # gradient w.r.t. X, shape = [m, d]
            G = D.jacobian_product(E)
            return E, G
        
        
        
        n_iters = 100
        batch_size = 50
        seq_len_a = 100
        seq_len_b = 100
        dims = 5
        
        from timeit import default_timer as timer
        t = 0
        for i in numba.prange(n_iters):
            for i in numba.prange(batch_size):
                
                X = np.random.random((seq_len_a, dims))
                Y = np.random.random((seq_len_b, dims))
                start = timer()
                E, G = fun(X, Y)
                t += timer() - start
            
        
        
        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        B = torch.rand((batch_size, seq_len_b, dims))
        sdtw_gpu = pysdtw.SoftDTW(use_cuda=True)

        t_gpu, _, _ = timed_run(n_iters, A.cuda(), B.cuda(), sdtw_gpu)

        print("PYSDTW:{:.2f} BLONDEL:{:.2f} RATIO:{:.2f}".format(t_gpu, t, t_gpu/t))
        
if __name__ == '__main__':
    unittest.main()
