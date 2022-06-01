from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import scale
except ImportError:
    scale = None

import numpy as np


def test_all(scale=scale):
    test_interface(scale)
    test_simple(scale)
    test_empty(scale)
    test_big(scale)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'scale')
    print_stats('numpy_pandas')


def test_interface(scale=scale):
    with ExceptionInterception():
        matrix = np.array([[0, 1, 2, 3], [4, 4, 5, 5]])

        result = scale(matrix)

        assert isinstance(result, np.ndarray), \
            "scale must return an ndarray"


def test_simple(scale=scale):
    with ExceptionInterception():
        matrix = np.array([[0, 1,  2,  3],
                           [4, 4,  5, 5]])

        expected = np.array([[-1., -1., -1., -1.],
                             [ 1.,  1.,  1.,  1.]])

        assert np.all(np.abs(expected - scale(matrix)) < 10 ** -8)


def test_empty(scale=scale):
    with ExceptionInterception():
        matrix = np.array([[]])

        expected = np.array([[]])

        assert np.all(expected == scale(matrix))


def test_big(scale=scale):
    with ExceptionInterception():
        matrix = np.array([[0, 1,  2,  3,  4.],
                           [5,  6,  7,  8, 9],
                           [10, 11, 12, 13, 14.],
                           [15, 16, 17, 18, 19.],
                           [20, 21, 22, 23, 24.]])
        expected = np.array([[-1.41421356, -1.41421356, -1.41421356, -1.41421356, -1.41421356],
                             [-0.70710678, -0.70710678, -0.70710678, -0.70710678, -0.70710678],
                             [0., 0., 0., 0., 0.],
                             [0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678],
                             [1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356]])

        assert np.all(np.abs(expected - scale(matrix)) < 10 ** -8)
