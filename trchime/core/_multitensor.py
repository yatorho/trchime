"""
The real implementions for tensor.
"""
import builtins
import warnings
from typing import List, Union, Iterable

from .tensor import Tensor
from .dependency import Dependency

import numpy as np


def _sum(t: 'Tensor', axis=None, keepdims: bool = False) -> 'Tensor':
    """
    Also see:
    -----------------
    trchime.multitensor.sum

    :param t:
    :param axis:
    :param keepdims:
    :return:
    """
    data: 'np.ndarray' = t.data.sum(axis = axis, keepdims = keepdims)

    requires_grad = t.requires_grad

    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':

            return grad * np.ones_like(t.data)

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]

        if not keepdims and data.shape != ():
            warnings.warn('assigned non-keepdims with requires-grad tensor', FutureWarning)

    else:
        depends_on: List['Dependency'] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _var(t: 'Tensor', axis, ddof: int, keepdims: bool) -> 'Tensor':
    """

    :param t:
    :param axis:
    :param ddof:
    :param keepdims:
    :return:
    """
    return _sum(_pow(t - _mean(t, axis, True), Tensor(2)), axis, keepdims) / (np.size(t.data, axis = axis) - ddof)


def _mean(t: 'Tensor', axis=None, keepdims: bool = False) -> 'Tensor':
    """
    Also See:
    ---------------
    trchime.multitensor.mean

    :param keepdims:
    :param axis:
    :param t:
    :return: Tensor(data,
                    requires_grad,
                    depends_on)
    """

    return _sum(t = t, axis = axis, keepdims = keepdims) / np.size(t.data, axis = axis)


def _add(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """
    Also See
    ----------
    trchime.multitensor.add

    :param t1:
    :param t2:
    :return:
    """

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # handle broadcasting properly
            # sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            # sum arcoss broadcasted  (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis = i, keepdims = True)

            return grad * np.ones_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis = i, keepdims = True)

            return grad * np.ones_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _mul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """
    Also see:
    ----------
    trchime.multitensor.mul

    :param t1:
    :param t2:
    :return:
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            grad = grad * t2.data

            # handle broadcasting properly
            # sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            # sum arcoss broadcasted  (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis = i, keepdims = True)
            return grad * np.ones_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis = i, keepdims = True)

            return grad * np.ones_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _pow(t: 'Tensor', power: 'Tensor') -> 'Tensor':
    """
    Also see:
    --------
    trchime.multitensor.pow
    :param t:
    :param power:
    :return:
    """
    data = np.power(t.data, power.data)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: 'np.ndarray') -> 'np.ndarray':
            grad = power.data * np.power(t.data, power.data - 1) * grad

            # handle broadcasting properly
            # sum out added dims
            ndims_added = grad.ndim - t.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            # sum arcoss broadcasted  (but non-added dims)
            for i, dim in enumerate(t.shape):
                if dim == 1:
                    grad = grad.sum(axis = i, keepdims = True)

            return grad * np.ones_like(t.data)

        depends_on = [Dependency(tensor = t, grad_fn = grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _rec(t: 'Tensor') -> 'Tensor':
    """
    Also see:
    --------
    trchime.multitensor.rec

    :param t:
    :return:
    """
    data = 1 / t.data
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_f(grad: np.ndarray) -> np.ndarray:
            return - grad * data * data

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _div(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """
    Also see:
    --------
    trchime.multitensor.rec

    :param t1:
    :param t2:
    :return:
    """
    return _mul(t1 = t1, t2 = _rec(t2))


def _neg(t: 'Tensor') -> 'Tensor':
    """

    Also see:
    --------
    trchime.multiensor.neg

    :param t:
    :return:
    """

    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            return -grad

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _sub(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """

    Also see:
    --------
    tchime.multitensor.sub

    :param t1:
    :param t2:
    :return:
    """
    return _add(t1, _neg(t2))


def _matmul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    trchime.multitensor.matmul

    :param t1:
    :param t2:
    :return:
    """

    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _dot(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """
    Also see
    --------
    tchime._multitensor.matmul
    :param t1:
    :param t2:
    :return:
    """
    return _matmul(t1 = t1, t2 = t2)


def _transpose_T(t: 'Tensor') -> 'Tensor':
    """
    Also see:
    --------
    trchime.multitensor.transpose_T
    trchime.multitensor.transpose
    :param t:
    :return:
    """
    data = t.data.T
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            return grad.T

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _transpose(t: 'Tensor', *axes) -> 'Tensor':
    """
    Also see:
    ---------
    trchime.multitensor.transpose
    trchime.multitensor.transpose_T
    trchime._multitensor.transpose
    trchime._multitensro.transpose_Y
    :param t:
    :param axes:
    :return:
    """
    data = t.data.transpose(*axes)
    requires_grad = t.requires_grad
    if requires_grad:

        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            if axes is not None:
                reve_list: List[int] = []
                for i in axes:
                    reve_list.append(i)

                reve_sequence: List[int] = []
                for n in range(grad.ndim):
                    reve_sequence.append(reve_list.index(n))

                return grad.transpose(tuple(reve_sequence))
            else:
                return grad.transpose()

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _slice(t: 'Tensor', idxs: slice or tuple, isnew: bool = True) -> 'Tensor':
    """
    Also see:
    --------
    trchime.multitensor.slice
    :param t:
    :param idxs:
    :param isnew:
    :return:
    """
    if isinstance(idxs, Tensor):
        idxs = idxs.data

    # ensure slice could be run as idxs  assigned as a tensor
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _max(t: 'Tensor', axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
    """
    Also see:
    --------

    :param t:
    :param axis:
    :param keepdims:
    :param isnew:
    :return:
    """
    data = t.data.max(axis = axis, keepdims = keepdims)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:

            grad = grad * (t.data >= data)
            return grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _min(t: 'Tensor', axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
    """

    :param t:
    :param axis:
    :param keepdims:
    :param isnew:
    :return:
    """
    data = t.data.min(axis = axis, keepdims = keepdims)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:

            grad = grad * (t.data <= data)
            return grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _argmax(t: 'Tensor', axis=None, isnew: bool = True) -> 'Tensor':
    """
    Also see
    -------

    :param t:
    :param axis:
    :param isnew:
    :return:
    """
    data = t.data.argmax(axis = axis)
    requires_grad = t.requires_grad

    if isnew:
        requires_grad = False

    if requires_grad:

        def grad_f(grad: np.ndarray) -> 'np.ndarray':

            return np.zeros_like(t.data)

        depends_on = [Dependency(t, grad_f)]
    else:
        depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def _argmin(t: 'Tensor', axis=None, isnew: bool = True) -> 'Tensor':
    """
    Also see:
    --------

    :param t:
    :param axis:
    :param isnew:
    :return:
    """
    data = t.data.argmin(axis = axis)
    requires_grad = t.requires_grad

    if isnew:
        requires_grad = False

    if requires_grad:

        def grad_f(grad: np.ndarray) -> 'np.ndarray':

            return np.zeros_like(t.data)

        depends_on = [Dependency(t, grad_f)]
    else:
        depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def _eq(t1: 'Tensor', t2: 'Tensor', isnew: bool = True) -> 'Tensor':
    """
    Also see
    ---------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    data = t1.data == t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _ne(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    -------
    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    data = t1.data != t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _gt(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    --------
    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    data = t1.data > t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _lt(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    data = t1.data < t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _ge(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """
    data = t1.data >= t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _le(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see:
    ---------

    :param t1:
    :param t2:
    :param isnew:
    :return:
    """

    data = t1.data <= t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: 'np.ndarray') -> 'np.ndarray':
            # Maually, discontinuous function just take its gradient to zero.
            return np.zeros_like(t1.data)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: 'np.ndarray') -> 'np.ndarray':
            return np.zeros_like(t2.data)

        depends_on.append(Dependency(t2, grad_fn2))

    if isnew:
        requires_grad = False
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _absolute_equal(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :return:
    """
    res = np.sum(t1.data == t2.data)

    if only_value:
        if res < t1.data.size:
            return False
        else:
            return True

    else:
        raise NotImplementedError


def _absolute_negequal(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data != t2.data)

    if only_value:
        if res < t1.data.size:
            return False
        else:
            return True
    else:
        raise NotImplementedError



def _absolute_gt(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> 'bool':
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data > t2.data)

    if only_value:
        if res < t1.data.size:
            return False
        else:
            return True
    else:
        raise NotImplementedError


def _absolute_lt(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data < t2.data)

    if only_value:
        if res < t1.data.size:
            return False
        else:
            return True
    else:
        raise NotImplementedError


def _absolute_ge(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data >= t2.data)

    if only_value:
        if res < t1.data.size:
            return False
        else:
            return True
    else:
        raise NotImplementedError


def _absolute_le(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data <= t2.data)

    if only_value:
        if res < t1.data. size:
            return False
        else:
            return True
    else:
        raise NotImplementedError

def _some_equal(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> 'bool':
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data == t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError


def _some_negequal(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> 'bool':
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data != t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError

def _some_gt(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data > t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError


def _some_ge(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data >= t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError

def _some_lt(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """

    res = np.sum(t1.data < t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError


def _some_le(t1: 'Tensor', t2: 'Tensor', only_value: bool) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    res = np.sum(t1.data <= t2.data)

    if only_value:
        if res > 0:
            return True
        else:
            return False
    else:
        raise NotImplementedError


def _reshape(t: 'Tensor', shape) -> 'Tensor':
    """
    Also see
    ---------

    :param t:
    :param shape:
    :return:
    """
    data = t.data.reshape(shape)

    requires_grad = t.requires_grad

    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':

            return grad.reshape(t.shape)

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _astype(t: 'Tensor', dtype: Union[str, object]) -> 'Tensor':
    """
    Also see
    ---------

    :param t:
    :param dtype:
    :return:
    """
    data = t.data.astype(dtype = dtype)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _copy(t: 'Tensor', only_data: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :param only_data:
    :return:
    """
    data = t.data.copy()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: 'np.ndarray') -> np.ndarray:
            return grad * np.ones_like(data)

        depends_on = [Dependency(tensor = t, grad_fn = grad_fn)]
    else:
        depends_on = []

    if only_data:
        requires_grad = False
        depends_on: List[Dependency] = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def _inv(t: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :param isnew:
    :return:
    """
    data = np.linalg.inv(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            raise NotImplementedError

        depends_on = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on = []

    if isnew:
        requires_grad = False
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _square(t: 'Tensor') -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :return:
    """
    return _pow(t = t, power = Tensor(2))


def _vstack(t1: 'Tensor', t2: 'Tensor', isnew: bool) -> 'Tensor':
    """
    Also see
    ----------
    :param t1:
    :param t2:
    :param isnew:
    :return:
    """

    data = np.vstack((t1.data, t2.data))
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_f1(grad: np.ndarray) -> 'np.ndarray':
            if t1.ndim == 0:
                return grad[0, 0]
            elif t1.ndim == 1:
                return (grad[0]).reshape(t1.shape[0])
            else:
                return grad[0: t1.shape[0]]

        depends_on.append(Dependency(tensor = t1, grad_fn = grad_f1))

    if t2.requires_grad:
        def grad_f2(grad: np.ndarray) -> 'np.ndarray':
            if t2.ndim == 0:
                return grad[1, 0]
            elif t2.ndim == 1:
                return (grad[1]).reshape(t2.shape[0])
            else:
                return grad[t1.shape[0]:]

        depends_on.append(Dependency(tensor = t2, grad_fn = grad_f2))

    if isnew:
        requires_grad = False
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _stack(t1: 'Tensor', t2: 'Tensor', axis: int, isnew: bool) -> 'Tensor':
    """
    Also see
    ---------

    :param t1:
    :param t2:
    :param axis:
    :param isnew:
    :return:
    """
    data = np.stack((t1.data, t2.data), axis = axis)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on = []

    if t1.requires_grad:

        def grad_f1(grad: np.ndarray) -> np.ndarray:

            idxs = []
            for dm in range(grad.ndim):
                if axis == dm:
                    idxs.append(0)
                else:
                    idxs.append(builtins.slice(None, None, None))
            grad = (grad[tuple(idxs)]).reshape(t1.data.shape)

            return grad

        depends_on.append(Dependency(tensor = t1, grad_fn = grad_f1))

    if t2.requires_grad:
        def grad_f2(grad: np.ndarray) -> np.ndarray:

            idxs = []
            for dm in range(grad.ndim):
                if axis == dm:
                    idxs.append(1)
                else:
                    idxs.append(builtins.slice(None, None, None))
            grad = (grad[tuple(idxs)]).reshape(t2.data.shape)

            return grad

        depends_on.append(Dependency(tensor = t2, grad_fn = grad_f2))

    if isnew:
        requires_grad = False
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _vstack_n(tup: Iterable['Tensor']) -> 'Tensor':
    raise NotImplementedError


def _size(t: 'Tensor', axis, isnew: bool) -> 'Tensor':
    """
    Also see
    --------

    :param t:
    :param axis:
    :param isnew
    :return:
    """

    data = np.size(t.data, axis)
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            # the gradient fucntion fo size op is just 0.
            return np.zeros_like(t.data)

        depends_on: List[Dependency] = [Dependency(tensor = t, grad_fn = grad_f)]

    else:
        depends_on: List[Dependency] = []

    if isnew:
        requires_grad = False
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)
