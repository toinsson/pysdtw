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
        import sdtw
        return True
