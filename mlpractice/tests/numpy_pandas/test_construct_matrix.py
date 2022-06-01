from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import construct_matrix
except ImportError:
    construct_matrix = None

import numpy as np


def test_all(construct_matrix=construct_matrix):
    test_interface(construct_matrix)
    test_simple(construct_matrix)
    test_empty(construct_matrix)
    test_big(construct_matrix)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'construct_matrix')
    print_stats('numpy_pandas')


def test_interface(construct_matrix=construct_matrix):
    with ExceptionInterception():
        first_array = np.array([1, 2, 3])
        second_array = np.array([4, 5, 6])

        result = construct_matrix(first_array, second_array)

        assert isinstance(result, np.ndarray), \
            "construct_matrix must return an ndarray"


def test_simple(construct_matrix=construct_matrix):
    with ExceptionInterception():
        first_array = np.array([1, 2, 3])
        second_array = np.array([4, 5, 6])

        expected = np.array([[1, 4], [2, 5], [3, 6]])

        assert np.all(expected == construct_matrix(first_array, second_array))


def test_empty(construct_matrix=construct_matrix):
    with ExceptionInterception():
        first_array=np.array([])
        second_array=np.array([])

        expected = np.array([]).reshape(0, 2)

        assert np.all(expected == construct_matrix(first_array, second_array))

def test_big(construct_matrix=construct_matrix):
    with ExceptionInterception():
        first_array = np.arange(0, 100, 2)
        second_array = np.arange(1, 100, 2)

        expected = np.arange(100).reshape(50, 2)

        assert np.all(expected == construct_matrix(first_array, second_array))
