from mlpractice.stats.stats_utils import _update_stats, print_stats

try:
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import GradientDescent
except ImportError:
    GradientDescent = None

import numpy as np
from mlpractice.utils import ExceptionInterception


def test_all(descent=GradientDescent):
    test_interface(descent)
    test_public(descent)
    test_default(descent)
    test_random(descent, 100)
    print('All tests passed!')
    _update_stats('gradient_descent', 'GradientDescent')
    print_stats('gradient_descent')


def test_interface(descent=GradientDescent):
    with ExceptionInterception():
        desc_1 = descent(np.array([[1.0], [1.0], [1.0]]), 1.5)
        desc_2 = descent(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), 3)

        # W
        assert hasattr(desc_1, 'W') and hasattr(desc_2, 'W'), \
            'in task you must store weights in the variable W'

        # calc_gradient
        assert hasattr(descent, 'calc_gradient'), \
            'descent must have method calc_gradient'
        assert callable(descent.calc_gradient), \
            'calc_gradient must be callable'

        gradient_1 = desc_1.calc_gradient(np.array([[1, 4, 8],
                                                    [1, 9, 27],
                                                    [1, 16, 64]]),
                                          np.array([[13.3], [15.6], [27.1]]))
        gradient_2 = desc_2.calc_gradient(np.array([[1, 2, 3, 0, 2],
                                                    [1, 0, 17, 3, 2],
                                                    [1, 11, -2, 7, 9]]),
                                          np.array([[100], [-12], [-17.538]]))
        assert isinstance(gradient_1, np.ndarray), \
            'calc_gradient must return np.ndarray'
        assert isinstance(gradient_2, np.ndarray), \
            'calc_gradient must return np.ndarray'
        assert gradient_1.shape == desc_1.W.shape, \
            'calc_gradient must return a np.ndarray with ' \
            'the same shape as descent weights'
        assert gradient_2.shape == desc_2.W.shape, \
            'calc_gradient must return a np.ndarray with ' \
            'the same shape as descent weights'

        # update_weights
        assert hasattr(descent, 'update_weights'), \
            'descent must have method update_weights'
        assert callable(descent.calc_gradient), \
            'update_weights must be callable'

        weight_diff_1 = desc_1.update_weights(gradient_1, 1)
        weight_diff_2 = desc_2.update_weights(gradient_2, 112)
        assert isinstance(weight_diff_1, np.ndarray), \
            'update_weights must return np.ndarray'
        assert isinstance(weight_diff_2, np.ndarray), \
            'update_weights must return np.ndarray'
        assert weight_diff_1.shape == desc_1.W.shape, \
            'update_weights must return a np.ndarray with ' \
            'the same shape as descent weights'
        assert weight_diff_2.shape == desc_2.W.shape, \
            'update_weights must return a np.ndarray with ' \
            'the same shape as descent weights'


def test_public(descent=GradientDescent):
    with ExceptionInterception():
        desc = descent(np.array([[1.0], [1.0], [1.0]]), 1.5)

        # calc_gradient
        grad = desc.calc_gradient(np.array([
            [1, 1, 1],
            [1, 2, 0.5],
            [1, 3, 1.0 / 3],
            [1, 4, 0.25]
        ]), np.array([[4], [3.5], [4], [4.75]]))
        expected_grad = np.array([[-1.0 / 12], [1.0], [-55.0 / 144]])
        assert np.allclose(grad, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        # update_weights
        diff = desc.update_weights(grad, 3)
        assert np.allclose(diff, 0.75 * expected_grad, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, np.array([[1.0], [1.0], [1.0]]) - diff, atol=10 ** -6), \
            'difference misuse noticed'


def test_default(descent=GradientDescent):
    with ExceptionInterception():
        input_W = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        input_lambda = 1.2843
        eta =  lambda k: input_lambda * (1 / (1 + k)) ** 0.5
        input_X = np.array([
            [1, 1, 1, 0, 2],
            [1, 2, 7, 1, 6],
            [1, 2, 9, 2, 2],
            [1, 3, 2, 1.5, 1.3],
            [1, 8, 4, 0.54, 0.23]
        ])
        input_Y = np.array([[1], [2], [3], [4], [5]])
        desc = descent(input_W, input_lambda)
        grad = desc.calc_gradient(input_X, input_Y)
        expected_grad = 2 * np.dot(input_X.T,
                                     (np.dot(input_X,
                                             input_W) - input_Y)) / input_X.shape[0]
        assert np.allclose(grad, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        diff = desc.update_weights(grad, 154)
        expected_diff = eta(154) * grad

        assert np.allclose(diff, expected_diff, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, input_W - diff, atol=10 ** -6), \
            'difference misuse noticed'


def test_random(descent=GradientDescent, iterations=1):
    with ExceptionInterception():
        np.random.seed(214)
        for test in range(iterations):
            inp_iteration = int(np.random.random() * 100) + 1
            inp_lambda = np.random.random() * 10
            samples_number = int(np.random.random() * 100) + 1
            weight_length = int(np.random.random() * 9) + 1
            inp_w0 = 20 * np.random.ranf((weight_length, 1)) - 10
            inp_X = 200 * np.random.ranf((samples_number, weight_length)) - 100
            inp_Y = 200 * np.random.ranf((samples_number, 1)) - 100
            eta = lambda k: inp_lambda * (1 / (1 + k)) ** 0.5

            desc = descent(inp_w0, inp_lambda)
            grad = desc.calc_gradient(inp_X, inp_Y)
            expected_grad = 2 * np.dot(inp_X.T,
                                       (np.dot(inp_X,
                                               inp_w0) - inp_Y)) / samples_number
            assert np.allclose(grad, expected_grad, atol=10 ** -6), \
                'wrong gradient received'

            diff = desc.update_weights(grad, inp_iteration)
            expected_diff = eta(inp_iteration) * grad
            assert np.allclose(diff, expected_diff, atol=10 ** -6), \
                'wrong difference calculated'
            assert np.allclose(desc.W, inp_w0 - diff, atol=10 ** -6), \
                'difference misuse noticed'
