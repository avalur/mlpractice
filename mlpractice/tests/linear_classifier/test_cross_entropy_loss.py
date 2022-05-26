from mlpractice.stats.stats_utils import \
    print_stats, _update_stats

try:
    from mlpractice_solutions.mlpractice_solutions.\
        linear_classifier_solution import cross_entropy_loss
except ImportError:
    cross_entropy_loss = None

from scipy.special import softmax
import torch
import numpy as np
from mlpractice.utils import ExceptionInterception


def test_all(cross_entropy_loss=cross_entropy_loss):
    test_interface(cross_entropy_loss)
    test_public(cross_entropy_loss)
    test_default(cross_entropy_loss)
    test_random(cross_entropy_loss, 100)
    print('All tests passed!')
    _update_stats('linear_classifier', 'cross_entropy_loss')
    print_stats('linear_classifier')


def test_interface(cross_entropy_loss=cross_entropy_loss):
    with ExceptionInterception():
        probs1 = np.array([0.1, 0.2, 0.7])
        target_index1 = np.array([2])
        probs2 = np.array([[0.1, 0.2, 0.7],
                           [0.1, 0.2, 0.7]])
        target_index2 = np.array([2, 2])

        loss1 = cross_entropy_loss(probs1, target_index1)
        loss2 = cross_entropy_loss(probs2, target_index2)

        assert isinstance(loss1, float), \
            "cross_entropy_loss must return a float"
        assert isinstance(loss2, float), \
            "cross_entropy_loss must return a float"


def test_public(cross_entropy_loss=cross_entropy_loss):
    with ExceptionInterception():
        probs = np.array([0.1, 0.2, 0.7])
        target_index = np.array([2])

        sample_answer = -np.log(0.7)

        assert abs(cross_entropy_loss(probs, target_index) - sample_answer) < \
               10 ** -6


def test_default(cross_entropy_loss=cross_entropy_loss):
    with ExceptionInterception():
        predictions = np.array([1, 2, 3])
        probs = softmax(predictions)  # [0.09003057, 0.24472847, 0.66524096]

        target_index = np.array([2])

        loss = torch.nn.CrossEntropyLoss(reduction='sum')

        sample_output = loss(
            torch.from_numpy(predictions[np.newaxis, :]).float(),
            torch.from_numpy(target_index).long(),
        )

        assert abs(cross_entropy_loss(probs, target_index) - sample_output) < \
               10 ** -6


def test_random(cross_entropy_loss=cross_entropy_loss, iterations=1):
    with ExceptionInterception():
        np.random.seed(42)

        for _ in range(iterations):
            predictions = np.random.rand(3, 4)
            probs = softmax(predictions, axis=1)

            target_index = np.random.randint(0, 4, size=3)

            loss = torch.nn.CrossEntropyLoss(reduction='sum')

            sample_output = loss(torch.from_numpy(predictions).float(),
                                 torch.from_numpy(target_index).long())

            assert abs(
                cross_entropy_loss(probs, target_index) - sample_output
            ) < 10 ** -6
