from mlpractice.stats.stats_utils import _update_stats, print_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.mlpractice_solutions\
        .linear_classifier_solution import linear_softmax
except ImportError:
    linear_softmax = None

import torch
import numpy as np


def test_all(linear_softmax=linear_softmax):
    test_interface(linear_softmax)
    test_public(linear_softmax)
    test_normalization(linear_softmax)
    test_random(linear_softmax, 100)
    print('All tests passed!')
    _update_stats('linear_classifier', 'linear_softmax')
    print_stats('linear_classifier')


def test_interface(linear_softmax=linear_softmax):
    with ExceptionInterception():
        objects1 = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        weights1 = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        target_index1 = np.array([0, 1, 2])
        loss1, gradient1 = linear_softmax(objects1, weights1, target_index1)

        objects2 = np.array([[1],
                             [2],
                             [3]])
        weights2 = np.array([[1, 2, 3]])
        target_index2 = np.array([0])
        loss2, gradient2 = linear_softmax(objects2, weights2, target_index2)

        assert isinstance(loss1, float), \
            "linear_softmax must return a float and an ndarray"
        assert isinstance(gradient1, np.ndarray), \
            "linear_softmax must return a float and an ndarray"
        assert gradient1.shape == weights1.shape, \
            "The output gradient shape must match the W shape"
        assert isinstance(loss2, float), \
            "linear_softmax must return a float and an ndarray"
        assert isinstance(gradient2, np.ndarray), \
            "linear_softmax must return a float and an ndarray"
        assert gradient2.shape == weights2.shape, \
            "The output gradient shape must match the W shape"


def test_public(linear_softmax=linear_softmax):
    with ExceptionInterception():
        objects = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        weights = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
        target_index = np.array([0, 1, 2])
        loss, gradient = linear_softmax(objects, weights, target_index)

        objects_tensor = torch.from_numpy(objects).float()
        weights_tensor = torch.from_numpy(weights).float()
        weights_tensor.requires_grad = True

        predictions_tensor = objects_tensor @ weights_tensor

        sample_loss = torch.nn.CrossEntropyLoss(reduction='sum')

        sample_output = sample_loss(predictions_tensor,
                                    torch.from_numpy(target_index).long())
        sample_output.backward()

        assert abs(loss - sample_output) < 10 ** -6
        assert np.all(np.abs(gradient - weights_tensor.grad.numpy()) <
                      10 ** -6)


def test_normalization(linear_softmax=linear_softmax):
    with ExceptionInterception():
        objects = np.array([[0, 0, 10000]])
        weights = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        target_index = np.array([2])
        loss, gradient = linear_softmax(objects, weights, target_index)

        objects_tensor = torch.from_numpy(objects).float()
        weights_tensor = torch.from_numpy(weights).float()
        weights_tensor.requires_grad = True

        predictions_tensor = objects_tensor @ weights_tensor

        sample_loss = torch.nn.CrossEntropyLoss(reduction='sum')

        sample_output = sample_loss(predictions_tensor,
                                    torch.from_numpy(target_index).long())
        sample_output.backward()

        assert abs(loss - sample_output) < 10 ** -6
        assert np.all(np.abs(gradient - weights_tensor.grad.numpy()) <
                      10 ** -6)


def test_random(linear_softmax=linear_softmax, iterations=1):
    with ExceptionInterception():
        np.random.seed(42)

        for _ in range(iterations):
            objects = np.random.rand(3, 3)
            weights = np.random.rand(3, 3)
            target_index = np.random.randint(0, 3, size=3)

            loss, gradient = linear_softmax(objects, weights, target_index)

            objects_tensor = torch.from_numpy(objects).float()
            weights_tensor = torch.from_numpy(weights).float()
            weights_tensor.requires_grad = True

            predictions_tensor = objects_tensor @ weights_tensor

            sample_loss = torch.nn.CrossEntropyLoss(reduction='sum')

            sample_output = sample_loss(predictions_tensor,
                                        torch.from_numpy(target_index).long())
            sample_output.backward()

            assert abs(loss - sample_output) < 10 ** -6
            assert np.all(np.abs(gradient - weights_tensor.grad.numpy()) <
                          10 ** -6)
