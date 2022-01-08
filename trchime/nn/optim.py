"""
Optimizers go here
"""

from typing import List

from ..core.initial.core import zeros_like
from ..core.multitensor import sqrt, square
from ..random import random
from ..core.tensor import Tensor
from ..core.module import Module


SGD_OPTIMIZER = 'StochasticGradientDescentOptimizer'
SGDM_OPTIMIZER = 'MomentumOptimizer'
ADAGRAD_OPTIMIZER = 'AdaptiveGradientOptimizer'
RMSPROP_OPTIMIZER = 'RootMeanSquarePropOptimizer'
ADAM_OPTIMIZER = 'AdaptiveMomentEstimationOptimizer'

class _SGD_imple:
    """ SGD optimizer """

    def __init__(self, lr: float) -> 'None':
        self.lr = lr

    def step(self, module: Module) -> 'None':
        for parameter in module.parameters():
            parameter.assign_sub(self.lr * parameter.grad)


class SGD(_SGD_imple):
    """

    """

    def __init__(self, lr: float = 0.01) -> 'None':
        if lr is None:
            lr = 0.01
        super().__init__(lr = lr)

    def step(self, module: Module) -> 'None':
        if not isinstance(module, Module):
            raise RuntimeError('optimizer get no module')
        super().step(module)


class _SGDM_imple:
    """ SGD with momentum """

    def __init__(self, lr: float, beta: float) -> 'None':
        self.lr = lr
        self.beta = beta
        self._isinit = False
        self.v_variable: List[Tensor] = []

    def _init_grad(self, module: Module) -> 'None':
        """

        :param module:
        :return:
        """
        for p in module.parameters():
            self.v_variable.append(zeros_like(p))

    def step(self, module: Module) -> 'None':
        """

        :param module:
        :return:
        """
        if not self._isinit:
            self._isinit = True
            self._init_grad(module)

        for n, parameter in enumerate(module.parameters()):
            self.v_variable[n] = self.beta * self.v_variable[n] + (1 - self.beta) * parameter.grad
            self.v_variable[n].non_gradient()

            parameter.assign_sub(self.lr * self.v_variable[n])


class SGDM(_SGDM_imple):
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        if lr is None:
            lr = 0.01
        if beta is None:
            beta = 0.9

        super().__init__(lr, beta)

    def step(self, module: Module):
        if not isinstance(module, Module):
            raise RuntimeError('optimizer get no module')

        super().step(module)


class _RMSprop_imple:
    """ RMSprop optimizer"""

    def __init__(self, lr: float, beta: float):
        self.lr = lr
        self.beta = beta

        self._isinit = False
        self.s_variable: List[Tensor] = []

    def _init_grad(self, module) -> 'None':
        for p in module.parameters():
            self.s_variable.append(zeros_like(p) + random(p.shape) * 1e-5)

    def step(self, module: Module) -> 'None':

        if not self._isinit:
            self._isinit = True
            self._init_grad(module)

        for n, parameter in enumerate(module.parameters()):
            self.s_variable[n] = self.beta * self.s_variable[n] + (1 - self.beta) * square(parameter.grad)
            self.s_variable[n].non_gradient()

            parameter.assign_sub(self.lr * parameter.grad / sqrt(self.s_variable[n]))


class RMSprop(_RMSprop_imple):
    """

    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        if lr is None:
            lr = 0.01
        if beta is None:
            beta = 0.9

        super().__init__(lr, beta)

    def step(self, module: Module):
        if not isinstance(module, Module):
            raise RuntimeError('optimizer get no module')

        super().step(module)


class _Adagrad_imple:
    """ Adagrad optimizer"""

    def __init__(self, lr: float):
        self.lr = lr

        self._isinit = False
        self.s_variable: List[Tensor] = []

    def _init_grad(self, module) -> 'None':
        for p in module.parameters():
            self.s_variable.append(zeros_like(p) + random(p.shape) * 1e-5)

    def step(self, module: Module) -> 'None':

        if not self._isinit:
            self._isinit = True
            self._init_grad(module)

        for n, parameter in enumerate(module.parameters()):
            self.s_variable[n] += square(parameter.grad)
            self.s_variable[n].non_gradient()

            parameter.assign_sub(self.lr * parameter.grad / sqrt(self.s_variable[n]))


class Adagrad(_Adagrad_imple):
    """

    """

    def __init__(self, lr: float = 0.01) -> 'None':
        if lr is None:
            lr = 0.01
        super().__init__(lr = lr)

    def step(self, module: Module) -> 'None':
        if not isinstance(module, Module):
            raise RuntimeError('optimizer get no module')
        super().step(module)


class _Adam_imple:
    """ Adam optimizer """

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.9, corrected: bool = False) -> 'None':
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.corrected = corrected

        self._step = 1
        self._isinit = False
        self.v_variable: List[Tensor] = []
        self.s_variable: List[Tensor] = []

    def _init_grad(self, module: Module) -> 'None':
        for p in module.parameters():
            self.v_variable.append(zeros_like(p) + random(p.shape) * 1e-5)
            self.s_variable.append(zeros_like(p) + random(p.shape) * 1e-5)

    def step(self, module: Module) -> 'None':
        if not self._isinit:
            self._isinit = True
            self._init_grad(module)

        for n, parameter in enumerate(module.parameters()):
            self.v_variable[n] = self.beta1 * self.v_variable[n] + (1 - self.beta1) * parameter.grad
            self.v_variable[n].non_gradient()

            self.s_variable[n] = self.beta2 * self.s_variable[n] + (1 - self.beta2) * square(parameter.grad)

            self.s_variable[n].non_gradient()

            if self.corrected:
                v_variable_corrected = self.v_variable[n] / (1 - self.beta1 ** self._step)
                s_variable_corrected = self.s_variable[n] / (1 - self.beta2 ** self._step)

                parameter.assign_sub(self.lr * v_variable_corrected / sqrt(s_variable_corrected))

                self._step += 1
            else:
                parameter.assign_sub(self.lr * self.v_variable[n] / sqrt(self.s_variable[n]))


class Adam(_Adam_imple):
    """

    """

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.9, corrected: bool = False) -> 'None':

        if lr is None:
            lr = 0.01
        if beta1 is None:
            beta1 = 0.9
        if beta2 is None:
            beta2 = 0.9
        if corrected is None:
            corrected = False

        super().__init__(lr = lr, beta1 = beta1, beta2 = beta2, corrected = corrected)

    def step(self, module: Module) -> 'None':
        if not isinstance(module, Module):
            raise RuntimeError('optimizer get no module')
        super().step(module)
