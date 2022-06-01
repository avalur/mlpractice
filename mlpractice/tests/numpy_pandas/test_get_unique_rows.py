from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import get_unique_rows
except ImportError:
    get_unique_rows = None

import numpy as np


def test_all(get_unique_rows=get_unique_rows):
    test_interface(get_unique_rows)
    test_simple(get_unique_rows)
    test_big(get_unique_rows)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'get_unique_rows')
    print_stats('numpy_pandas')


def test_interface(get_unique_rows=get_unique_rows):
    with ExceptionInterception():
        matrix = np.array([[1, 2, 3]])

        result = get_unique_rows(matrix)

        assert isinstance(result, np.ndarray), \
            "get_unique_rows must return an ndarray"


def test_simple(get_unique_rows=get_unique_rows):
    with ExceptionInterception():
        with ExceptionInterception():
            matrix = np.array([[1, 2, 3]])

            expected = np.array([[1, 2, 3]])

        assert np.all(expected == get_unique_rows(matrix))


def test_big(get_unique_rows=get_unique_rows):
    with ExceptionInterception():
        matrix = np.array([[4, 5, 6],
                           [0, 1, 2],
                           [1, 2, 3],
                           [0, 1, 2],
                           [4, 5, 6],
                           [1, 2, 3]]),

        expected = np.array([[0, 1, 2],
                             [1, 2, 3],
                             [4, 5, 6]])

        assert np.all(expected == get_unique_rows(matrix))
