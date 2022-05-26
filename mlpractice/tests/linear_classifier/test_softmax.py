from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.linear_classifier_solution import softmax
except ImportError:
    softmax = None

from scipy.special import softmax as softmax_sample
import numpy as np


def test_all(softmax=softmax):
    test_interface(softmax)
    test_public(softmax)
    test_default(softmax)
    test_normalization(softmax)
    test_random(softmax, 100)
    print('All tests passed!')
    _update_stats('linear_classifier', 'softmax')
    print_stats('linear_classifier')


def test_interface(softmax=softmax):
    with ExceptionInterception():
        x1 = np.array([1, 2, 3])
        x2 = np.array([[1, 2, 3],
                       [1, 2, 3]])

        y1 = softmax(x1)
        y2 = softmax(x2)

        assert isinstance(y1, np.ndarray), \
            "softmax must return an ndarray"
        assert x1.shape == y1.shape, \
            "The output shape must match the input shape"
        assert isinstance(y2, np.ndarray), \
            "softmax must return an ndarray"
        assert x2.shape == y2.shape, \
            "The output shape must match the input shape"


def test_public(softmax=softmax):
    with ExceptionInterception():
        x = np.array([1, 2, 3])

        y_sample = softmax_sample(x)
        y = softmax(x)

        assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_default(softmax=softmax):
    with ExceptionInterception():
        x = np.array([[1, 0.5, 0.2, 3],
                      [1, -1, 7, 3],
                      [2, 12, 13, 3]])

        y_sample = softmax_sample(x, axis=1)
        y = softmax(x)

        assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_normalization(softmax=softmax):
    with ExceptionInterception():
        x = np.array([10000, 0, 0])

        y_sample = softmax_sample(x)
        y = softmax(x)

        assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_random(softmax=softmax, iterations=1):
    with ExceptionInterception():
        np.random.seed(42)

        for _ in range(iterations):
            x = np.random.rand(3, 4)

            y_sample = softmax_sample(x, axis=1)
            y = softmax(x)

            assert np.all(np.abs(y - y_sample) < 10 ** -8)
