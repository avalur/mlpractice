import numpy as np
from typing import Union

class BaseValues:
    S0_default: float = 1
    P_default: float = 0.5
    batch_size_default: int = 1
    alpha_default: float = 0.1
    eps_default: float = 1e-8
    mu_default = 1e-2
    tolerance_default: float = 1e-3
    max_iter_default: int = 1000


class BaseDescent:
    r"""A base class and examples for all functions

    Attributes
    ----------
    W : np.ndarray
        Weights.
    """

    def __init__(self):
        self.W = None

    def step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        r"""Descenting step

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        iteration: int
            Iteration number
        """
        return self.update_weights(self.calc_gradient(X, y), iteration)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        pass

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        pass


class GradientDescent(BaseDescent):
    r"""Gradient descent class.

    Attributes
    ----------
    W : np.ndarray
        Weights.
    """
    def __init__(self, W0: np.ndarray, lambda_: float,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        W0: np.ndarray
            Initialize weights.
        lambda_: float
            Learning rate parameter (step scale)
        S0: float
            Learning rate parameter
        P: float
            Learning rate parameter
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (S0 / (S0 + k)) ** P
        self.W = np.copy(W0)
    
    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        # TODO: Implement updating weights
        raise NotImplementedError('Not implemented!')
    
    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: Implement calculating MSE gradient
        raise NotImplementedError('Not implemented!')


class StochasticDescent(BaseDescent):
    r"""Stochastic gradient descent class

    Attributes
    ----------
    W : np.ndarray
        Weights.
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default,
                 batch_size: int = BaseValues.batch_size_default):
        r"""
        Parameters
        ----------
        W0: np.ndarray
            Initialize weights.
        lambda_: float
            Learning rate parameter (step scale)
        S0: float
            Learning rate parameter
        P: float
            Learning rate parameter
        batch_size: int
            Batch size
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (S0 / (S0 + k)) ** P
        self.batch_size = batch_size
        self.w = np.copy(W0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        # TODO: implement updating weights function
        raise NotImplementedError('Not implemented!')

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: implement calculating gradient function
        raise NotImplementedError('Not implemented!')


class MomentumDescent(BaseDescent):
    r"""Momentum gradient descent class

    Attributes
    ----------
    W : np.ndarray
        Weights.
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 alpha: float = BaseValues.alpha_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        W0: np.ndarray
            Initialize weights.
        lambda_: float
            Learning rate parameter (step scale)
        alpha: float
            Momentum coefficient
        S0: float
            Learning rate parameter
        P: float
            Learning rate parameter
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (S0 / (S0 + k)) ** P
        self.alpha = alpha
        self.w = np.copy(W0)
        self.h = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        # TODO: implement updating weights function
        raise NotImplementedError('Not implemented!')

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: implement calculating gradient function
        raise NotImplementedError('Not implemented!')


class Adagrad(BaseDescent):
    r"""Adaptive gradient algorithm class

    Attributes
    ----------
    W : np.ndarray
        Weights.
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 eps: float = BaseValues.eps_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        W0: np.ndarray
            Initialize weights.
        lambda_: float
            Learning rate parameter (step scale)
        eps: float
            Smoothing term
        S0: float
            Learning rate parameter
        P: float
            Learning rate parameter
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (S0 / (S0 + k)) ** P
        self.eps = eps
        self.w = np.copy(W0)
        self.g = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        raise NotImplementedError('Not implemented!')

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: implement calculating gradient function
        raise NotImplementedError('Not implemented!')


class GradientDescentReg(GradientDescent):
    r"""Full gradient descent with regularization class

    Attributes
    ----------
    mu: float
        l2 regularization coefficient
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 mu: float = BaseValues.mu_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        mu: float
            l2 regularization coefficient
        """
        super().__init__(W0=W0, lambda_=lambda_, S0=S0, P=P)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: calculate l2
        l2 = None
        raise NotImplementedError('Not implemented!')
        return super().calc_gradient(X, Y) + l2 * self.mu


class StochasticDescentReg(StochasticDescent):
    r"""Stochastic gradient descent with regularization class

    Attributes
    ----------
    mu: float
        l2 regularization coefficient
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 mu: float = BaseValues.mu_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default,
                 batch_size: int = BaseValues.batch_size_default):
        r"""
        Parameters
        ----------
        mu: float
            l2 regularization coefficient
        """
        super().__init__(W0=W0, lambda_=lambda_, S0=S0, P=P, batch_size=batch_size)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: calculate l2
        l2 = None
        raise NotImplementedError('Not implemented!')
        return super().calc_gradient(X, Y) + l2 * self.mu


class MomentumDescentReg(MomentumDescent):
    r"""Momentum gradient descent with regularization class
    Attributes
    ----------
    mu: float
        l2 regularization coefficient
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 alpha: float = BaseValues.alpha_default,
                 mu: float = BaseValues.mu_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        mu: float
            l2 regularization coefficient
        """
        super().__init__(W0=W0, lambda_=lambda_, alpha=alpha, S0=S0, P=P)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Calculating MSE gradient

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        gradient: np.ndarray
            Calculating gradient
        """
        # TODO: calculate l2
        l2 = None
        raise NotImplementedError('Not implemented!')
        return super().calc_gradient(X, Y) + l2 * self.mu


class AdagradReg(Adagrad):
    r"""Adaptive gradient algorithm with regularization class

    Attributes
    ----------
    mu: float
        l2 regularization coefficient
    """

    def __init__(self, W0: np.ndarray, lambda_: float,
                 eps: float = BaseValues.eps_default,
                 mu: float = BaseValues.mu_default,
                 S0: float = BaseValues.S0_default,
                 P: float = BaseValues.P_default):
        r"""
        Parameters
        ----------
        mu: float
            l2 regularization coefficient
        """
        super().__init__(W0=W0, lambda_=lambda_, eps=eps, S0=S0, P=P)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        r"""
        Parameters
        ----------
        W0: np.ndarray
            Initialize weights.
        lambda_: float
            Learning rate parameter (step scale)
        alpha: float
            Momentum coefficient
        S0: float
            Learning rate parameter
        P: float
            Learning rate parameter
        """
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r"""Changing weights with respect to gradient

        Parameters
        ----------
        gradient: np.ndarray
            Gradient of MSE
        iteration: int
            Iteration number

        Returns
        -------
        weigh_diff : np.ndarray
            Weight difference
        """
        # TODO: calculate l2
        l2 = None
        raise NotImplementedError('Not implemented!')
        return super().calc_gradient(X, Y) + l2 * self.mu


class LinearRegression:
    r"""
    Linear regression class

    Attributes
    ----------
    descent: Union[GradientDescent, StochasticDescent, MomentumDescent, Adagrad,
                   GradientDescentReg, StochasticDescentReg, MomentumDescentReg, AdagradReg]?
        Descent class    
    tolerance: float
        Stopping criterion for square of euclidean norm of weight difference
    max_iter: int
        Stopping criterion for iterations    
    loss_history
        Progress history
    """

    def __init__(self, descent: Union[GradientDescent, StochasticDescent, MomentumDescent, Adagrad,
                                      GradientDescentReg, StochasticDescentReg, MomentumDescentReg, AdagradReg],
                 tolerance: float = BaseValues.tolerance_default,
                 max_iter: int = BaseValues.max_iter_default):
        r"""
        Parameters
        ----------
        descent: Union[GradientDescent, StochasticDescent, MomentumDescent, Adagrad,
                       GradientDescentReg, StochasticDescentReg, MomentumDescentReg, AdagradReg]?
            Descent class    
        tolerance: float
            Stopping criterion for square of euclidean norm of weight difference
        max_iter: int
            Stopping criterion for iterations    
        """
        self.descent = descent
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'LinearRegression':
        r"""Getting objects, fitting descent weights

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        
        Returns
        -------
        self: LinearRegression
        """
        # TODO: fit weights to X and Y
        raise NotImplementedError('Not implemented!')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        r"""Getting objects, predicting targets

        Parameters
        ----------
        X: np.ndarray
            Features

        Returns
        -------
        Y: np.ndarray
            Predicted targets
        """
        # TODO: calculate prediction for X
        raise NotImplementedError('Not implemented!')

    def calc_loss(self, X: np.ndarray, Y: np.ndarray) -> None:
        r"""Getting objects, calculating loss

        Parameters
        ----------
        X: np.ndarray
            Features
        Y: np.ndarray
            Targets
        """
        # TODO: calculate loss and save it to loss_history
        raise NotImplementedError('Not implemented!')
