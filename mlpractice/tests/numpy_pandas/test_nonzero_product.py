from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import nonzero_product
except ImportError:
    nonzero_product = None

import numpy as np


def test_all(nonzero_product=nonzero_product):
    test_interface(nonzero_product)
    test_simple(nonzero_product)
    test_empty(nonzero_product)
    test_zeros(nonzero_product)
    test_big(nonzero_product)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'nonzero_product')
    print_stats('numpy_pandas')


def test_interface(nonzero_product=nonzero_product):
    with ExceptionInterception():
        matrix = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])

        result = nonzero_product(matrix)

        assert isinstance(result, float), \
            "nonzero_product must return a float"


def test_simple(nonzero_product=nonzero_product):
    with ExceptionInterception():
        matrix = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]])

        expected = 3.0

        assert expected == nonzero_product(matrix)


def test_empty(nonzero_product=nonzero_product):
    with ExceptionInterception():
        matrix = np.array([[]])

        assert nonzero_product(matrix) is None


def test_zeros(nonzero_product=nonzero_product):
    with ExceptionInterception():
        matrix = np.array([[0, 0, 1], [2, 0, 2], [3, 0, 0], [4, 4, 4]])

        assert nonzero_product(matrix) is None


def test_big(nonzero_product=nonzero_product):
    with ExceptionInterception():
        matrix = np.arange(100).reshape(10, 10)

        expected = 855652058110080

        assert expected == nonzero_product(matrix)
