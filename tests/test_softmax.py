from mlpractice_solutions.mlpractice_solutions.linear_classifier_solution\
    import softmax

from scipy.special import softmax as softmax_sample
import numpy as np


def test_all():
    test_public()
    test_default()
    test_normalization()
    test_random(100)


def test_public():
    x = np.array([1, 2, 3])

    y_sample = softmax_sample(x)
    y = softmax(x)

    assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_default():
    x = np.array([[1, 0.5, 0.2, 3],
                  [1, -1, 7, 3],
                  [2, 12, 13, 3]])

    y_sample = softmax_sample(x, axis=1)
    y = softmax(x)

    assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_normalization():
    x = np.array([10000, 0, 0])

    y_sample = softmax_sample(x)
    y = softmax(x)

    assert np.all(np.abs(y - y_sample) < 10 ** -8)


def test_random(iterations=1):
    np.random.seed(42)

    for _ in range(iterations):
        x = np.random.rand(3, 4)

        y_sample = softmax_sample(x, axis=1)
        y = softmax(x)

        assert np.all(np.abs(y - y_sample) < 10 ** -8)
