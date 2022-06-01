from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import replace_nans
except ImportError:
    replace_nans = None

import numpy as np
from numpy import nan


def test_all(replace_nans=replace_nans):
    test_interface(replace_nans)
    test_simple(replace_nans)
    test_all_nan(replace_nans)
    test_empty(replace_nans)
    test_big(replace_nans)
    test_no_nan(replace_nans)
    test_one_nan(replace_nans)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'replace_nans')
    print_stats('numpy_pandas')


def test_interface(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.array([[nan, 1, 2, 3], [4, nan, 5, nan]])

        result = replace_nans(matrix)

        assert isinstance(result, np.ndarray), \
            "replace_nans must return an ndarray"


def test_simple(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.array([[nan, 1, 2, 3], [4, nan, 5, nan]])

        expected = np.array([[3, 1, 2, 3], [4, 3, 5, 3]])

        assert np.all(expected == replace_nans(matrix))


def test_all_nan(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.ones((3, 14)) * nan

        expected = np.zeros((3, 14))

        assert np.all(expected == replace_nans(matrix))


def test_empty(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.array([[]])

        expected = np.array([[]])

        assert np.all(expected == replace_nans(matrix))


def test_big(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.array([[0, nan,  2,  3,  4.],
                           [5,  6,  7,  8, nan],
                           [nan, 11, 12, 13, 14.],
                           [15, 16, 17, nan, 19.],
                           [20, 21, nan, 23, 24.]])
        expected = np.array([[0, 12,  2,  3,  4.],
                             [5,  6,  7,  8, 12.],
                             [12, 11, 12, 13, 14.],
                             [15, 16, 17, 12, 19.],
                             [20, 21, 12, 23, 24.]])

        assert np.all(expected == replace_nans(matrix))


def test_no_nan(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix = np.array([[3]])

        expected = np.array([[3]])

        assert np.all(expected == replace_nans(matrix))


def test_one_nan(replace_nans=replace_nans):
    with ExceptionInterception():
        matrix=np.array([[1, nan]])

        expected = np.array([[1, 1]])

        assert np.all(expected == replace_nans(matrix))
