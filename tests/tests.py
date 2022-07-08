import torch
import unittest

# run with:
# python -m unittest
# python -m unittest tests.tests.TestCaseName.FunctionName

class TestImport(unittest.TestCase):

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

class TestLegacy(unittest.TestCase):

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


    def test_equal_legacy_distance(self):
        import sys
        sys.path.append("tests")
        import soft_dtw_cuda
        import pysdtw

        batch_size, seq_len_a, seq_len_b, dims = 10, 512, 1023, 15

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

        # forward
        res_leg = sdtw_leg(A.cuda(), B.cuda())
        res = sdtw(Ac.cuda(), B.cuda())
        assert torch.allclose(res_leg, res)

class TestCompute(unittest.TestCase):

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

    def test_packed_a(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 10
        dims = 5
        
        # can process different input types
        len_a = torch.randint(10, 100, (batch_size,))
        a = [torch.rand((l, dims)) for l in len_a]
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b = torch.rand((batch_size, 25, dims))

        f10 = sdtw_cuda(a_packed.cuda(), b.cuda())
        f01 = sdtw_cuda(b.cuda(), a_packed.cuda())
        f11 = sdtw_cuda(a_packed.cuda(), a_packed.cuda())

    def test_packed_b(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 10
        dims = 5
        # can process different input types similarly

        # here a and a_packed are equal
        a = [torch.rand((35, dims)) for l in range(batch_size)]
        a = torch.stack(a)
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b = torch.rand((batch_size, 25, dims))

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        f0 = sdtw_cuda(a.cuda(), b.cuda())
        f1 = sdtw_cuda(a_packed.cuda(), b.cuda())
        torch.allclose(f0, f1)

    def test_packed_c(self):
        import pysdtw
        import torch.nn.utils.rnn as rnn

        sdtw_cuda = pysdtw.SoftDTW(use_cuda=True)
        batch_size = 3
        dims = 5
        
        # here a,b have different lengths
        len_a = torch.randint(10, 100, (batch_size,))
        len_b = torch.randint(10, 100, (batch_size,))
        a = [torch.rand((l, dims)) for l in len_a]
        b = [torch.rand((l, dims)) for l in len_b]

        res0 = torch.cat([sdtw_cuda(ai.unsqueeze(0).cuda(), bi.unsqueeze(0).cuda()) for (ai, bi) in zip(a, b)])
        a_packed = rnn.pack_sequence(a, enforce_sorted=False)
        b_packed = rnn.pack_sequence(b, enforce_sorted=False)
        res1 = sdtw_cuda(a_packed.cuda(), b_packed.cuda())
        
        torch.allclose(res0, res1)

if __name__ == '__main__':
    unittest.main()
