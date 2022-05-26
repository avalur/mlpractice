try:
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import MomentumDescentReg
except ImportError:
    MomentumDescentReg = None

import numpy as np
from mlpractice.utils import ExceptionInterception


def test_all(descent=MomentumDescentReg):
    test_interface(descent)
    test_public(descent)
    test_default(descent)
    test_random(descent, 100)
    print('All tests passed!')


def test_interface(descent=MomentumDescentReg):
    with ExceptionInterception():
        desc_1 = descent(np.array([[2.0], [8.0], [7.0]]), 1.52345353)
        desc_2 = descent(np.array([[13.0], [72.0], [33.0], [45.0], [555.0]]), 332)

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


def test_public(descent=MomentumDescentReg):
    with ExceptionInterception():
        desc = descent(np.array([[1.0], [1.0], [1.0]]), 1.5, alpha=1, mu=0.5)
        inp_x = np.array([
            [1, 1, 1],
            [1, 2, 0.5],
            [1, 3, 1.0 / 3],
            [1, 4, 0.25]
        ])
        inp_y = np.array([[4], [3.5], [4], [4.75]])

        # calc_gradient
        grad_1 = desc.calc_gradient(inp_x, inp_y)
        expected_grad = np.array([[5.0 / 12], [1.5], [17.0 / 144]])
        assert np.allclose(grad_1, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        # update_weights
        diff_1 = desc.update_weights(grad_1, 3)
        assert np.allclose(diff_1, 0.75 * expected_grad, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, np.array([[1.0], [1.0], [1.0]]) - diff_1, atol=10 ** -6), \
            'difference misuse noticed'

        # pornography starts here
        grad = desc.calc_gradient(inp_x, inp_y)
        expected_grad = np.array([[-28025.0 / 4608], [-1697.0 / 96], [-141821.0 / 55296]])
        assert np.allclose(grad, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        diff = desc.update_weights(grad, 3)
        expected_diff = np.array([[-26105.0 / 6144], [-1553.0 / 128], [-135293.0 / 73728]])
        assert np.allclose(diff, expected_diff, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, np.array([[1.0], [1.0], [1.0]]) - diff_1 - diff, atol=10 ** -6), \
            'difference misuse noticed'


def test_default(descent=MomentumDescentReg):
    with ExceptionInterception():
        input_W = np.array([[1.2307], [0.320534], [-2.760745], [1.53453], [0.0145004634013]])
        input_lambda = 1.2843
        input_mu = 0.00598546234
        eta =  lambda k: input_lambda * (1 / (1 + k)) ** 0.5
        input_X = np.array([
            [1, 1, 1, 0, 2],
            [1, 2, 7, 1, 6],
            [1, 2, 9, 2, 2],
            [1, 3, 2, 1.5, 1.3],
            [1, 8, 4, 0.54, 0.23]
        ])
        input_Y = np.array([[1], [2], [3], [4], [5]])
        desc = descent(input_W, input_lambda, mu=input_mu)
        grad_1 = desc.calc_gradient(input_X, input_Y)
        expected_grad = 2 * np.dot(input_X.T,
                                   (np.dot(input_X,
                                           input_W) - input_Y)) / input_X.shape[0] + input_mu * input_W
        assert np.allclose(grad_1, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        diff_1 = desc.update_weights(grad_1, 154)
        expected_diff = eta(154) * grad_1

        assert np.allclose(diff_1, expected_diff, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, input_W - diff_1, atol=10 ** -6), \
            'difference misuse noticed'

        grad_2 = desc.calc_gradient(input_X, input_Y)
        expected_grad = 2 * np.dot(input_X.T,
                                   (np.dot(input_X,
                                           desc.W) - input_Y)) / input_X.shape[0] + input_mu * (input_W - diff_1)
        assert np.allclose(grad_2, expected_grad, atol=10 ** -6), \
            'wrong gradient received'

        diff_2 = desc.update_weights(grad_2, 155)
        expected_diff = 0.1 * expected_diff + eta(155) * grad_2
        assert np.allclose(diff_2, expected_diff, atol=10 ** -6), \
            'wrong difference calculated'
        assert np.allclose(desc.W, input_W - diff_1 - diff_2, atol=10 ** -6), \
            'difference misuse noticed'


def test_random(descent=MomentumDescentReg, iterations=1):
    with ExceptionInterception():
        np.random.seed(214)
        for test in range(iterations):
            inp_iteration = int(np.random.random() * 100) + 1
            inp_lambda = np.random.random() * 10
            inp_mu = (np.random.random() + 10 ** -2) * (10 ** - 4)
            samples_number = int(np.random.random() * 100) + 1
            weight_length = int(np.random.random() * 9) + 1
            inp_w0 = 20 * np.random.ranf((weight_length, 1)) - 10
            inp_X = 200 * np.random.ranf((samples_number, weight_length)) - 100
            inp_Y = 200 * np.random.ranf((samples_number, 1)) - 100
            eta = lambda k: inp_lambda * (1 / (1 + k)) ** 0.5

            desc = descent(inp_w0, inp_lambda, mu=inp_mu)
            # h = 0
            grad = desc.calc_gradient(inp_X, inp_Y)
            expected_grad = 2 * np.dot(inp_X.T,
                                       (np.dot(inp_X,
                                               inp_w0) - inp_Y)) / samples_number + inp_mu * inp_w0
            assert np.allclose(grad, expected_grad, atol=10 ** -6), \
                'wrong gradient received'

            diff = desc.update_weights(grad, inp_iteration)
            expected_diff = eta(inp_iteration) * grad
            assert np.allclose(diff, expected_diff, atol=10 ** -6), \
                'wrong difference calculated'
            assert np.allclose(desc.W, inp_w0 - diff, atol=10 ** -6), \
                'difference misuse noticed'

            # h changed

            grad_2 = desc.calc_gradient(inp_X, inp_Y)
            expected_grad = 2 * np.dot(inp_X.T,
                                       (np.dot(inp_X,
                                               desc.W) - inp_Y)) / samples_number + inp_mu * (inp_w0 - diff)
            assert np.allclose(grad_2, expected_grad, atol=10 ** -6), \
                'wrong gradient received'

            diff_2 = desc.update_weights(grad_2, inp_iteration + 1)
            expected_diff = 0.1 * diff + eta(inp_iteration + 1) * grad_2
            assert np.allclose(diff_2, expected_diff, atol=10 ** -6), \
                'wrong difference calculated'
            assert np.allclose(desc.W, inp_w0 - diff - diff_2, atol=10 ** -6), \
                'difference misuse noticed'
