"""
The real implementions math fucntion for mathematic and machine learning.
"""
from typing import List

from ._multitensor import _ge, _gt, _le, _lt
from .tensor import Tensor
from .dependency import Dependency
import numpy as np


def _tanh(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    :param t:
    :return:
    """
    data = np.tanh(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _sin(t: 'Tensor') -> 'Tensor':
    """
    Also see:
    ---------
    :param t:
    :return:
    """
    data = np.sin(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.cos(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _cos(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    :param t:
    :return:
    """
    data = np.cos(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return - grad * np.sin(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _tan(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    :param t:
    :return:
    """
    data = np.tan(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / (np.cos(t.data) * np.cos(t.data))

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _exp(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    :param t:
    :return:
    """

    data = np.exp(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _log(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :return:
    """
    data = np.log(t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / t.data

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _sigmoid(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :return:
    """

    data = 1 / (1 + np.exp(-t.data))
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _ReLU(t: 'Tensor') -> 'Tensor':
    """
    Also see
    ---------

    :param t:
    :return:
    """
    data = np.maximum(0, t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return 1 * (t.data >= 0) * grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _Leak_ReLU(t: 'Tensor', k: float) -> 'Tensor':
    """
    Also see
    ----------

    :param t:
    :param k:
    :return:
    """
    data = np.maximum(k * t.data, t.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # divide gradarray into two parts, one for 1, and other for k
            arr1 = 1 * (t.data >= 0) * grad
            arr2 = k * (t.data < 0) * grad

            return arr1 + arr2

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _softplus(t: 'Tensor') -> 'Tensor':
    """
    Also see
    ---------
    :param t:
    :return:
    """
    raise NotImplementedError


def _softmax(t: 'Tensor', axis) -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :param axis:
    :return:
    """
    return _exp(t) / (_exp(t).sum(axis = axis, keepdims = True))


def _maximum(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """

    res = t1 * _ge(t1, t2, isnew = True) + t2 * _gt(t2, t1, isnew = True)

    if isnew:
        res._depends_on = []
        res._requires_grad = False
        res._grad = None
    return res


def _minimum(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    ---------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    res = t1 * _le(t1, t2, isnew = True) + t2 * _lt(t2, t1, isnew = True)

    if isnew:
        res._depends_on = []
        res._requires_grad = False
        res._grad = None

    return res


def _one_hot(t: 'Tensor', depth, on_value, off_value, isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :param depth:
    :param on_value:
    :param off_value:
    :param isnew:
    :return:
    """
    pos_p = on_value * np.eye(depth)[t.data]
    neg_p = off_value * (np.zeros_like(pos_p) == pos_p)
    data = pos_p + neg_p

    requires_grad = t.requires_grad

    if requires_grad:
        def grad_f(grad: np.ndarray) -> np.ndarray:
            return np.zeros_like(t.data)

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _abs(t: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    ---------

    :param t:
    :param isnew:
    :return:
    """
    return _maximum(t, -t, isnew = isnew)
