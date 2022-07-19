import torch
import unittest

# run with:
# python -m unittest
# python -m unittest tests.tests.TestCaseName.FunctionName

class TestImport(unittest.TestCase):
    """Test importing the package needed for the testcases:
    - pysdtw: this package
    - soft_dtw_cuda: the package from which pysdtw is inspired
    - SoftDTW: Blondel original package
    """
    def test_import(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(use_cuda=False)
        sdtw = pysdtw.SoftDTW(use_cuda=True)
        return True

    def test_import_legacy(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        return True

    def test_import_blondel(self):
        import SoftDTW
        return True


def assert_forward_backward(x, A, y, Ac):
    """Assert that DTW discrepencies x and y, as well as gradients on A and Ac are equal.
    """
    assert torch.allclose(x, y)

    x_loss = x.sum()
    y_loss = y.sum()
    x_loss.backward()
    y_loss.backward()

    assert torch.allclose(A.grad, Ac.grad, atol=1e-2)


class TestLegacy(unittest.TestCase):
    """Test whether pysdtw is equivalent to soft_dtw_cuda.
    """
    def setUp(self):
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    def test_equal_legacy_cpu(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 5, 3, 12

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))

        sdtw_leg = soft_dtw_cuda.SoftDTW(False, gamma=1.0)
        sdtw = pysdtw.SoftDTW(gamma=1.0, use_cuda=False)

        res_leg = sdtw_leg(A, B)
        res = sdtw(Ac, B)

        assert_forward_backward(res_leg, A, res, Ac)

    def test_equal_legacy_gpu(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 5, 3, 12

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))

        sdtw_leg = soft_dtw_cuda.SoftDTW(True, gamma=1.0)
        sdtw = pysdtw.SoftDTW(gamma=1.0)

        res_leg = sdtw_leg(A.cuda(), B.cuda())
        res = sdtw(Ac.cuda(), B.cuda())

        assert_forward_backward(res_leg, A, res, Ac)

    def test_equal_legacy_distance(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 513, 259, 15

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))

        sdtw_leg = soft_dtw_cuda.SoftDTW(True, gamma=1.0)

        def pairwise_l2_squared(x, y):
            x_norm = (x**2).sum(-1).unsqueeze(-1)
            y_norm = (y**2).sum(-1).unsqueeze(-2)
            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.mT)
            return torch.clamp(dist, 0.0, torch.inf)
        sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=pairwise_l2_squared)

        res_leg = sdtw_leg(A.cuda(), B.cuda())
        res = sdtw(Ac.cuda(), B.cuda())

        assert_forward_backward(res_leg, A, res, Ac)


class TestCompute(unittest.TestCase):
    def setUp(self):
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    def test_equal_gpu_cpu(self):
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 11, 32, 57, 2
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.detach().clone().requires_grad_(True)
        b_gpu = b_cpu.cuda()

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)

        forward_cpu = sdtw_cpu(a_cpu, b_cpu)
        forward_gpu = sdtw_cuda(a_gpu.cuda(), b_gpu)

        assert_forward_backward(forward_cpu, a_cpu, forward_gpu.cpu(), a_gpu)

    def test_packed_a(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 10
        dims = 5
        
        len_a = torch.randint(10, 100, (batch_size,))
        a = [torch.rand((l, dims)) for l in len_a]
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b = torch.rand((batch_size, 25, dims))

        f10 = sdtw_cuda(a_packed.cuda(), b.cuda())
        f01 = sdtw_cuda(b.cuda(), a_packed.cuda())
        f11 = sdtw_cuda(a_packed.cuda(), a_packed.cuda())
        
        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)
        f10 = sdtw_cpu(a_packed, b)
        f01 = sdtw_cpu(b, a_packed)
        f11 = sdtw_cpu(a_packed, a_packed)
        
    def test_packed_b(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 10
        dims = 5

        A0 = [torch.rand((35, dims)) for l in range(batch_size)]
        A = torch.stack(A0).requires_grad_(True)
        Ac = A.detach().clone().requires_grad_(True)
        Ac_packed = rnn.pack_sequence(Ac, enforce_sorted=False)
        B = torch.rand((batch_size, 25, dims))

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        f0 = sdtw_cuda(A.cuda(), B.cuda())
        f1 = sdtw_cuda(Ac_packed.cuda(), B.cuda())
        assert_forward_backward(f0, A, f1, Ac)

    def test_packed_c(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 10
        dims = 5

        A0 = [torch.rand((35, dims)) for l in range(batch_size)]
        A = torch.stack(A0).requires_grad_(True)
        Ac = A.detach().clone().requires_grad_(True)
        Ac_packed = rnn.pack_sequence(Ac, enforce_sorted=False)
        B = torch.rand((batch_size, 25, dims))

        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)
        f0 = sdtw_cpu(A, B)
        f1 = sdtw_cpu(Ac_packed, B)
        assert_forward_backward(f0, A, f1, Ac)

    def test_packed_d(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 3
        dims = 5
        
        len_a = torch.randint(10, 100, (batch_size,))
        len_b = torch.randint(10, 100, (batch_size,))
        a = [torch.rand((l, dims), requires_grad=True) for l in len_a]
        b = [torch.rand((l, dims)) for l in len_b]

        res0 = torch.cat([sdtw_cuda(ai.unsqueeze(0).cuda(), bi.unsqueeze(0).cuda()) for (ai, bi) in zip(a, b)])
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b_packed = rnn.pack_sequence(b, enforce_sorted=False)
        res1 = sdtw_cuda(a_packed.cuda(), b_packed.cuda())
        
        torch.allclose(res0, res1)

    def test_packed_e(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn
        
        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)
        batch_size = 20
        dims = 5

        len_a = torch.randint(20, 50, (batch_size,))
        len_b = torch.randint(30, 50, (batch_size,))
        a = [torch.rand((l, dims), requires_grad=True) for l in len_a]
        b = [torch.rand((l, dims)) for l in len_b]

        # ground truth, DTW computed separately on the batch
        res0 = torch.cat([sdtw_cuda(ai.unsqueeze(0).cuda(), bi.unsqueeze(0).cuda()) 
                          for (ai, bi) in zip(a, b)])
        loss0 = res0.sum()
        loss0.backward()
        grad0 = torch.cat([ai.grad for ai in a])

        # test GPU
        a_c0 = [ai.detach().clone().requires_grad_(True) for ai in a]
        a_packed = rnn.pack_sequence(a_c0, enforce_sorted=False)
        b_packed = rnn.pack_sequence(b, enforce_sorted=False)
        res1 = sdtw_cuda(a_packed.cuda(), b_packed.cuda())
        loss1 = res1.sum()
        loss1.backward()
        grad1 = torch.cat([ai.grad for ai in a_c0])
        assert torch.allclose(grad0, grad1, atol=1e-4)
        
        # test CPU
        a_c1 = [ai.detach().clone().requires_grad_(True) for ai in a]
        a_packed = rnn.pack_sequence(a_c1, enforce_sorted=False)
        b_packed = rnn.pack_sequence(b, enforce_sorted=False)
        res2 = sdtw_cpu(a_packed, b_packed)
        loss2 = res2.sum()
        loss2.backward()
        grad2 = torch.cat([ai.grad for ai in a_c1])
        assert torch.allclose(grad0, grad2, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
