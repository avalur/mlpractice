from mlpractice.stats.stats_utils import _update_stats, print_stats

try:
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import LinearRegression
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import GradientDescent
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import MomentumDescent
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import StochasticDescent
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import Adagrad
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import GradientDescentReg
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import MomentumDescentReg
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import StochasticDescentReg
    from mlpractice_solutions.mlpractice_solutions.gradient_descent_solutions import AdagradReg
except ImportError:
    LinearRegression = None
    GradientDescent = None
    MomentumDescent = None
    StochasticDescent = None
    Adagrad = None
    GradientDescentReg = None
    MomentumDescentReg = None
    StochasticDescentReg = None
    AdagradReg = None

import numpy as np
import math as mt
from mlpractice.utils import ExceptionInterception


def test_all(regression=LinearRegression, descents=tuple([GradientDescent, StochasticDescent,
                                                          MomentumDescent, Adagrad,
                                                          GradientDescentReg, StochasticDescentReg,
                                                          MomentumDescentReg, AdagradReg])):
    test_interface(regression)
    test_public(regression)
    test_default(regression)
    test_random(regression, descents, 5)
    print('All tests passed!')
    _update_stats('gradient_descent', 'LinearRegression')
    print_stats('gradient_descent')


def test_interface(regression=LinearRegression, descents=tuple([GradientDescent, StochasticDescent,
                                                                MomentumDescent, Adagrad,
                                                                GradientDescentReg, StochasticDescentReg,
                                                                MomentumDescentReg, AdagradReg])):
    with ExceptionInterception():
        assert hasattr(regression, 'fit'), \
            'linear regression must have method fit'

        assert callable(regression.fit), \
            'fit must be callable'

        assert hasattr(regression, 'predict'), \
            'linear regression must have method predict'

        assert callable(regression.predict), \
            'predict must be callable'

        for descent in descents:
            reg = regression(descent(np.array([[0.4], [1.3], [2.2], [13.3]]), 0.5), 10 ** -4, 100)
            assert hasattr(reg, 'loss_history'), \
                'linear regression must have attribute loss_history'

            assert isinstance(reg.fit(np.array([[1, 1, 10, 1],
                              [1, 2, 10 ** 2, 1.0 / 2],
                              [1, 3, 10 ** 3, 1.0 / 3]]),
                    np.array([[12],
                              [112],
                              [1112]])
                    ), regression), \
                'fit must return linear regression'

            res = reg.predict(np.array([[1, 0.5, 10 ** 0.5, 2],
                                        [1, 0.25, 10 * 0.25, 4],
                                        [1, -1, 10 ** -1, -1],
                                        [1, -2, 10 ** -2, -0.5],
                                        [1, -4, 10 ** -4, -0.25],
                                        [1, -0.5, 10 ** -0.5, -2]])
                              )

            assert isinstance(res, np.ndarray), \
                'predict must return ndarray'

            assert res.shape == (6, 1), \
                'predict must return as many answers as there are batches given'


def test_public(regression=LinearRegression, descents=tuple([GradientDescent, StochasticDescent,
                                                             MomentumDescent, Adagrad,
                                                             GradientDescentReg, StochasticDescentReg,
                                                             MomentumDescentReg, AdagradReg])):
    with ExceptionInterception():
        np.random.seed(12)
        bgen = lambda x: [1.0, x, x ** 2]
        to_interpolate = lambda x: [3.0 + 2.0 * x - 0.5 * (x ** 2)]
        for descent in descents:
            reg = regression(descent(np.array([
                [1.0],
                [1.0],
                [1.0]
            ]), 0.0067), 10 ** -6, 550)
            reg.fit(np.array([
                bgen(1),
                bgen(2),
                bgen(3),
                bgen(4),
                bgen(5)
            ]),
                np.array([
                    to_interpolate(1),
                    to_interpolate(2),
                    to_interpolate(3),
                    to_interpolate(4),
                    to_interpolate(5)
                ]))
            assert np.allclose(reg.predict(np.array([
                bgen(0.5),
                bgen(1.5),
                bgen(2.5),
                bgen(3.5),
                bgen(4.5),
            ])),
                np.array([
                    to_interpolate(0.5),
                    to_interpolate(1.5),
                    to_interpolate(2.5),
                    to_interpolate(3.5),
                    to_interpolate(4.5),
                ]),
                atol=30
            ), \
                'interpolation results are wrong'


def test_default(regression=LinearRegression, descents=tuple([GradientDescent, StochasticDescent,
                                                              MomentumDescent, Adagrad,
                                                              GradientDescentReg, StochasticDescentReg,
                                                              MomentumDescentReg, AdagradReg])):
    with ExceptionInterception():
        np.random.seed(12)
        bgen = lambda x: [1, 1 / x, mt.floor(x), mt.sin(x), mt.cosh(x), x * mt.log(x)]
        to_interpolate = lambda x: [mt.sqrt(3) - mt.acos(0.5462546) / x +
                                    13 * mt.floor(x) + (35.0 / 613) * mt.sin(x) -
                                    mt.cosh(x) + 0.00001 * x * mt.log(x)]
        for descent in descents:
            reg = regression(descent(np.array([
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0]
            ]), 0.000067), 10 ** -6, 1350)
            reg.fit(np.array([
                bgen(1),        bgen(0.1),
                bgen(2),        bgen(0.2),
                bgen(3),        bgen(0.3),
                bgen(4),        bgen(0.4),
                bgen(5),        bgen(0.5),
                bgen(6),        bgen(0.6),
                bgen(0.11),     bgen(0.21),
                bgen(0.12),     bgen(0.22),
                bgen(0.13),     bgen(0.23),
                bgen(0.14),     bgen(0.24),
                bgen(0.15),     bgen(0.25),
                bgen(0.16),     bgen(0.26)
            ]),
                np.array([
                    to_interpolate(1),      to_interpolate(0.1),
                    to_interpolate(2),      to_interpolate(0.2),
                    to_interpolate(3),      to_interpolate(0.3),
                    to_interpolate(4),      to_interpolate(0.4),
                    to_interpolate(5),      to_interpolate(0.5),
                    to_interpolate(6),      to_interpolate(0.6),
                    to_interpolate(0.11),   to_interpolate(0.21),
                    to_interpolate(0.12),   to_interpolate(0.22),
                    to_interpolate(0.13),   to_interpolate(0.23),
                    to_interpolate(0.14),   to_interpolate(0.24),
                    to_interpolate(0.15),   to_interpolate(0.25),
                    to_interpolate(0.16),   to_interpolate(0.26)
                ])
            )
            assert np.allclose(reg.predict(np.array([
                bgen(0.131), bgen(0.12), bgen(0.43921), bgen(1.1),
                bgen(0.132), bgen(0.22), bgen(0.43922), bgen(1.2),
                bgen(0.133), bgen(0.32), bgen(0.43923), bgen(1.3),
                bgen(0.134), bgen(0.42), bgen(0.43924), bgen(1.4),
                bgen(0.135), bgen(0.52), bgen(0.43925), bgen(1.5),
                bgen(0.136), bgen(0.62), bgen(0.43926), bgen(1.6),
                bgen(0.137), bgen(0.72), bgen(0.43927), bgen(1.7),
                bgen(0.138), bgen(0.82), bgen(0.43928), bgen(1.8),
                bgen(0.139), bgen(0.92), bgen(0.43929), bgen(1.9)
            ])),
                np.array([
                    to_interpolate(0.131), to_interpolate(0.12), to_interpolate(0.43921), to_interpolate(1.1),
                    to_interpolate(0.132), to_interpolate(0.22), to_interpolate(0.43922), to_interpolate(1.2),
                    to_interpolate(0.133), to_interpolate(0.32), to_interpolate(0.43923), to_interpolate(1.3),
                    to_interpolate(0.134), to_interpolate(0.42), to_interpolate(0.43924), to_interpolate(1.4),
                    to_interpolate(0.135), to_interpolate(0.52), to_interpolate(0.43925), to_interpolate(1.5),
                    to_interpolate(0.136), to_interpolate(0.62), to_interpolate(0.43926), to_interpolate(1.6),
                    to_interpolate(0.137), to_interpolate(0.72), to_interpolate(0.43927), to_interpolate(1.7),
                    to_interpolate(0.138), to_interpolate(0.82), to_interpolate(0.43928), to_interpolate(1.8),
                    to_interpolate(0.139), to_interpolate(0.92), to_interpolate(0.43929), to_interpolate(1.9)
                ]),
                rtol=10
            ), \
                'interpolation results are wrong with descent'


def test_random(regression=LinearRegression, descents=tuple([GradientDescent, StochasticDescent,
                                                             MomentumDescent, Adagrad,
                                                             GradientDescentReg, StochasticDescentReg,
                                                             MomentumDescentReg, AdagradReg]),
                iterations=1):
    with ExceptionInterception():
        funcs = [mt.exp, mt.log, mt.sqrt, mt.atan, mt.sinh,
                 mt.cosh, mt.tanh, mt.sin, mt.cos, mt.ceil,
                 mt.floor, mt.degrees, mt.radians]
        np.random.seed(252346)
        for _ in range(iterations):
            for descent in descents:
                batch = np.random.randint(13, size=7)
                input_w0 = np.random.ranf((7, 1))
                inp_lambda = 0.000067
                samples_number = int((np.random.random() + 1) * 40)
                bgen = lambda x: [funcs[i](x) for i in batch]
                consts = np.random.ranf((13, )) * (10 ** -2)
                to_interpolate = lambda x: [ sum([consts[i] * funcs[i](x) for i in batch]) ]
                reg = regression(descent(input_w0, inp_lambda), 10 ** -6, 1300)
                inp_arg = 2 * np.random.ranf((samples_number, ))
                reg.fit(np.array([bgen(x) for x in inp_arg]), np.array([to_interpolate(x) for x in inp_arg]))
                arg_to_check = 2 * (np.random.ranf((20, )) + 1)
                assert np.allclose(reg.predict(np.array([bgen(x) for x in arg_to_check])),
                                np.array([to_interpolate(x) for x in arg_to_check]), rtol=1000), \
                    'interpolation results are wrong'
