import torch
import unittest

# run with:
# python -m unittest
# python -m unittest tests.tests.TestCaseName.FunctionName

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



if __name__ == '__main__':
    unittest.main()
