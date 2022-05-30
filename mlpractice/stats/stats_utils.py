import inspect
import json
import os
import requests
from typing import List, Union, Dict

import mlpractice


_stats_path = os.path.join(
    os.path.dirname(inspect.getfile(mlpractice.stats)),
    'stats.json',
)


def _init_stats() -> dict:
    """Initialize a starting user's stats file."""
    stats = {
        'linear_classifier': {
            'softmax': 0,
            'cross_entropy_loss': 0,
            'softmax_with_cross_entropy': 0,
            'l2_regularization': 0,
            'linear_softmax': 0,
            'LinearSoftmaxClassifier': 0,
        },
        'rnn_torch': {
            'make_token_to_id': 0,
            'make_tokens': 0,
        },
        'gradient_descent': {
            'GradientDescent': 0,
            'StochasticDescent': 0,
            'MomentumDescent': 0,
            'Adagrad': 0,
            'GradientDescentReg': 0,
            'StochasticDescentReg': 0,
            'MomentumDescentReg': 0,
            'AdagradReg': 0,
            'LinearRegression': 0,
        },
    }
    with open(_stats_path, 'w') as stats_file:
        json.dump(stats, stats_file)
    return stats


# TODO: this might become a bottleneck in case we want to track multiple users
def _get_stats() -> Dict[str, Dict[str, int]]:
    """Reads the user's stats file and returns its content."""
    try:
        with open(_stats_path, 'r') as stats_file:
            stats = json.load(stats_file)
    except FileNotFoundError:
        raise RuntimeError(
            'In order to get stats, one should initialize them first! '
            'Run `mlpractice init` to initialize directory with tasks.'
        )

    return stats


def print_stats(hw_name: str = '', task_name: str = ''):
    """Prints user's progress on completing the course tasks.

    Parameters
    ----------
    hw_name : str, optional
        A course homework about which the statistics was requested.
    task_name : str, optional
        A homework task about which the statistics was requested.
    """
    stats = _get_stats()

    if not hw_name and not task_name:
        for hw_name in stats:
            print(f'{hw_name}: {_decorate(list(stats[hw_name].values()))}')
            for task_name in stats[hw_name]:
                print(f' - {task_name}: {_decorate(stats[hw_name][task_name])}')
        return

    if hw_name:
        if hw_name not in stats:
            raise ValueError(f'Invalid homework name was provided: {hw_name}!')

        if task_name:
            if task_name not in stats[hw_name]:
                raise ValueError(
                    f'Invalid task name was provided: {task_name}!'
                )

            print(
                f'{hw_name} - {task_name}:',
                _decorate(stats[hw_name][task_name]),
            )
            return

        print(f'{hw_name}: {_decorate(list(stats[hw_name].values()))}')
        for task_name in stats[hw_name]:
            print(f' - {task_name}: {_decorate(stats[hw_name][task_name])}')
        return

    if task_name:
        for hw_name in stats:
            if task_name in stats[hw_name]:
                print(
                    f'{hw_name} - {task_name}:',
                    _decorate(stats[hw_name][task_name]),
                )
                return

        raise ValueError(f'Invalid task name was provided: {task_name}!')


def _decorate(stats_record: Union[List[int], int]):
    """Makes a decorated string representation of `stats_query`.

    Parameters
    ----------
    stats_record : int or list of int
        Record from user's stats file.
    """
    if isinstance(stats_record, int):
        return '🟩' if stats_record else '🟥'
    if isinstance(stats_record, list):
        return ' '.join(
            ['🟩' if result else '🟥' for result in stats_record]
        )
    raise ValueError('An int or list of int expected!')


def _update_stats(hw_name: str, task_name: str):
    """Updates the user's stats file."""
    stats = _get_stats()
    stats[hw_name][task_name] = 1
    with open(_stats_path, 'w') as stats_file:
        json.dump(stats, stats_file)


def submit(username, password, stats):
    """Submits solutions to server"""
    r = requests.post(
        "SERVER",
        data={
            "username": username,
            "password": password,
            "stats": stats
        }
    )
    html = r.text
    possible_error = html[html.find('<title>') + 7: html.find(' //')]
    if possible_error.startswith('ValueError'):
        print(possible_error)
    else:
        print("Successfully submitted!")
