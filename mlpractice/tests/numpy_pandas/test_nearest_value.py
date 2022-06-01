from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import nearest_value
except ImportError:
    nearest_value = None

import numpy as np


def test_all(nearest_value=nearest_value):
    test_interface(nearest_value)
    test_simple(nearest_value)
    test_empty(nearest_value)
    test_big(nearest_value)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'nearest_value')
    print_stats('numpy_pandas')


def test_interface(nearest_value=nearest_value):
    with ExceptionInterception():
        matrix = np.arange(0, 10).reshape((2, 5))
        value = 3.6

        result = nearest_value(matrix, value)

        assert isinstance(result, float), \
            "construct_matrix must return a float"


def test_simple(nearest_value=nearest_value):
    with ExceptionInterception():
        matrix = np.arange(0, 10).reshape((2, 5))
        value = 3.6

        expected = 4.0

        assert expected == nearest_value(matrix, value)


def test_empty(nearest_value=nearest_value):
    with ExceptionInterception():
        matrix = np.array([[]])
        value = 0.0

        assert nearest_value(matrix, value) is None


def test_big(nearest_value=nearest_value):
    with ExceptionInterception():
        matrix = np.ones((100, 100))
        value = 1e6

        expected = 1.0

        assert expected == nearest_value(matrix, value)
