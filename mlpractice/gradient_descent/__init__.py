r"""This is the third (?) task of the mlpractice package.

You should fill in the gaps in the given function templates.
"""
from .gradient_descent import (
    BaseDescent,
    GradientDescent,
    StochasticDescent,
    MomentumDescent,
    Adagrad,
    GradientDescentReg,
    StochasticDescentReg,
    MomentumDescentReg,
    AdagradReg,
    LinearRegression
)

__all__ = [
    "BaseDescent",
    "GradientDescent",
    "StochasticDescent",
    "MomentumDescent",
    "Adagrad",
    "GradientDescentReg",
    "StochasticDescentReg",
    "MomentumDescentReg",
    "AdagradReg",
    "LinearRegression"
]
