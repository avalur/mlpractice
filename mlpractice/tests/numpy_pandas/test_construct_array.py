from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import construct_array
except ImportError:
    construct_array = None

import numpy as np


def test_all(construct_array=construct_array):
    test_interface(construct_array)
    test_diag(construct_array)
    test_simple(construct_array)
    test_empty(construct_array)
    test_row(construct_array)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'construct_array')
    print_stats('numpy_pandas')


def test_interface(construct_array=construct_array):
    with ExceptionInterception():
        matrix = np.array([[1, 2, 3],
                       [1, 2, 3]])
        row_indices = [0, 0]
        col_indices = [0, 2]

        result = construct_array(matrix, row_indices, col_indices)

        assert isinstance(result, np.ndarray), \
            "construct_array must return an ndarray"


def test_diag(construct_array=construct_array):
    with ExceptionInterception():
        matrix = np.array(range(25)).reshape(5, 5)
        row_indices = [0, 1, 2]
        col_indices = [0, 1, 2]

        expected = np.array([0, 6, 12])

        assert np.all(expected == construct_array(matrix, row_indices, col_indices))


def test_simple(construct_array=construct_array):
    with ExceptionInterception():
        matrix = np.arange(-10, 10).reshape((5, 4))
        row_indices = [1, 2, 3, 3]
        col_indices = [3, 2, 1, 2]

        expected = np.array([-3, 0, 3, 4])

        assert np.all(expected == construct_array(matrix, row_indices, col_indices))


def test_empty(construct_array=construct_array):
    with ExceptionInterception():
        matrix = np.arange(42).reshape((7, 6))
        row_indices = []
        col_indices = []

        expected = np.array([])

        assert np.all(expected == construct_array(matrix, row_indices, col_indices))


def test_row(construct_array=construct_array):
    with ExceptionInterception():
        matrix = np.arange(42).reshape((42, 1))
        row_indices = [0, 1, 41]
        col_indices = [0, 0, 0]

        expected = np.array([0, 1, 41])

        assert np.all(expected == construct_array(matrix, row_indices, col_indices))
