import torch
import unittest

# run with:
# python -m unittest 
# python -m unittest tests.tests.TestCase.test_equal_legacy_gpu

class TestCase(unittest.TestCase):

    def test_import(self):
        import pysdtw
        return True

    def test_import_cpu(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(False)
        return True

    def test_import_cuda(self):
        import pysdtw
        sdtw = pysdtw.SoftDTW(True)
        return True

    def test_import_legacy(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        return True

    def test_equal_legacy_gpu(self):
        """Test whether the pysdtw produces the same results both in forward and
        backward as the legacy code.
        """
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 1, 1400, 1500, 1

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))

        sdtw_leg = soft_dtw_cuda.SoftDTW(True, gamma=1.0, normalize=False)
        sdtw = pysdtw.SoftDTW(gamma=1.0, normalize=False)

        # forward
        res_leg = sdtw_leg(A.cuda(), B.cuda())
        res = sdtw(Ac.cuda(), B.cuda())
        assert torch.allclose(res_leg, res)

        # backward
        loss_leg = res_leg.sum()
        loss = res.sum()
        loss_leg.backward()
        loss.backward()
        assert torch.allclose(A.grad, Ac.grad, atol=1e-2)

        return True

    def test_equal_legacy_distance(self):
        """Tests equality for a new distance function.
        """
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda

        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 512, 1023, 15

        A = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        Ac = A.detach().clone().requires_grad_(True)
        B = torch.rand((batch_size, seq_len_b, dims))

        sdtw_leg = soft_dtw_cuda.SoftDTW(True, gamma=1.0, normalize=False)

        def pairwise_l2_squared(x, y):
            x_norm = (x**2).sum(-1).unsqueeze(-1)
            y_norm = (y**2).sum(-1).unsqueeze(-2)
            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.mT)
            return torch.clamp(dist, 0.0, torch.inf)

        sdtw = pysdtw.SoftDTW(gamma=1.0, normalize=False, dist_func=pairwise_l2_squared)

        # forward
        res_leg = sdtw_leg(A.cuda(), B.cuda())
        res = sdtw(Ac.cuda(), B.cuda())
        assert torch.allclose(res_leg, res)

    def test_equal_gpu_cpu(self):
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 512, 1023, 15
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        sdtw_cpu = pysdtw.SoftDTW(use_cuda=False)

        forward_cpu = sdtw_cpu(a_cpu, b_cpu)
        forward_gpu = sdtw_cuda(a_gpu, b_gpu)

        assert torch.allclose(forward_cpu, forward_gpu.cpu())


if __name__ == '__main__':
    unittest.main()
