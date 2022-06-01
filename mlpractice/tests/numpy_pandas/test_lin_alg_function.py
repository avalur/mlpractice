from mlpractice.stats.stats_utils import print_stats, _update_stats
from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.\
        mlpractice_solutions.numpy_pandas_solution import lin_alg_function
except ImportError:
    lin_alg_function = None

import numpy as np


def test_all(lin_alg_function=lin_alg_function):
    test_interface(lin_alg_function)
    test_random(lin_alg_function)
    print('All tests passed!')
    _update_stats('numpy_pandas', 'lin_alg_function')
    print_stats('numpy_pandas')


def test_interface(lin_alg_function=lin_alg_function):
    with ExceptionInterception():
        x = np.array([[0, 1], [4, 4]])

        result = lin_alg_function(x)

        assert isinstance(result, tuple), \
            "lin_alg_function must return a tuple"


def test_random(lin_alg_function=lin_alg_function):
    with ExceptionInterception():
        x = np.random.normal(size=(10,10))

        expected = (np.linalg.det(x),
                    np.trace(x),
                    np.amin(x), np.amax(x),
                    np.linalg.norm(x),
                    np.linalg.eigvals(x),
                    np.linalg.inv(x)
                    )
        actual = lin_alg_function(x)
        result = [np.all(e - a < 10 ** -8) for e, a in zip (expected, actual)]

        assert np.all(result)
