from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import max_element_spec
except ImportError:
    max_element_spec = None

import numpy as np


def test_all(max_element_spec=max_element_spec):
    test_interface(max_element_spec)
    test_simple(max_element_spec)
    test_empty1(max_element_spec)
    test_empty2(max_element_spec)
    test_empty3(max_element_spec)
    test_big(max_element_spec)
    test_in_a_row(max_element_spec)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'max_element_spec')
    print_stats('numpy_pandas')


def test_interface(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([1, 0, 2, 3])

        result = max_element_spec(x)

        assert isinstance(result, float), \
            "max_element_spec must return a float"


def test_simple(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])

        expected = 5.0

        assert expected == max_element_spec(x)


def test_empty1(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([])

        assert max_element_spec(x) is None


def test_empty2(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([6, 6])

        assert max_element_spec(x) is None


def test_empty3(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([1])

        assert max_element_spec(x) is None


def test_zeros(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.zeros(3)

        expected = 0.0

        assert expected == max_element_spec(x)


def test_big(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([6, 2, 0, 3, 0, 0, 9, 4])

        expected = 9

        assert expected == max_element_spec(x)


def test_in_a_row(max_element_spec=max_element_spec):
    with ExceptionInterception():
        x = np.array([1, 0, 0, -1])

        expected = 0.0

        assert expected == max_element_spec(x)
