"""

There is modle for mathematical fuctions and they gradient for
tensor's operation.

Some common functions had been recorded such as:
sin(x), cos(x), tan(x), tanh(x), exp(x), log(x)...

also, here implements useful functions for deep learning:
sigmoid(x), ReLU(x), softmax(x)...

This moudle would be imported to the AutoDiffer package, which
means you don't need to import it dividually.

"""
from ._func import _tanh, _sin, _cos, _tan, _exp, _log, _sigmoid, _ReLU, _Leak_ReLU, _softplus, _softmax, _maximum, \
    _minimum, _one_hot, _abs
from .tensor import Tensor

from .gather import Tensorable

# noinspection PyProtectedMember
from .gather import _ensure_tensor



def tanh(t: 'Tensorable') -> 'Tensor':
    """
    Implement tanh function and its gradient function:
    if puts x -> tanh(x), the gradient of tanh(x) would be
    (1 - tanh(x) * tanh(x)).

    :param t:
    :return: Tensor(data,
                   requires_grad,
                   depends_on)
    """
    t = _ensure_tensor(t)

    return _tanh(t = t)


def sin(t: 'Tensorable') -> 'Tensor':
    """
    Implement sine function and its gradient function:
    if puts x -> sin(x), the gradient of sin(x) would be
    cos(x).

    :param t:
    :return: Tensor(data,
                 requires_grad,
                 depends_on)
    """

    t = _ensure_tensor(t)

    return _sin(t)


def cos(t: 'Tensorable') -> 'Tensor':
    """
    Implement cosine function and its gradient function:
    if puts x -> cos(x), the gradient of cos(x) would be
    - sin(x).

    :param t:
    :return: Tensor(data,
                 requires_grad,
                 depends_on)
    """
    t = _ensure_tensor(t)

    return _cos(t = t)


def tan(t: 'Tensorable') -> 'Tensor':
    """
    Implement tan function and its gradient function:
    if puts x -> tan(x), the gradient of tan(x) would be
    1/(cos(x) * cos(x)).

    :param t:
    :return: Tensor(data,
                 requires_grad,
                 depends_on)
    """
    t = _ensure_tensor(t)

    return _tan(t = t)


def exp(t: 'Tensorable') -> 'Tensor':
    """
    Implement exp function and its gradient function:
    if puts x -> exp(x), the gradient of exp(x) would be
    exp(x).

    :param t:
    :return: Tensor(data,
                 requires_grad,
                 depends_on)
    """
    t = _ensure_tensor(t)

    return _exp(t = t)


def log(t: 'Tensorable') -> 'Tensor':
    """
    Implement log function and its gradient function:
    if puts x -> log(x), the gradient of log(x) would be
    1/x.

    :param t:
    :return: Tensor(data,
                 requires_grad,
                 depends_on)
    """
    t = _ensure_tensor(t)

    return _log(t = t)


def sigmoid(t: 'Tensorable') -> 'Tensor':
    """
    Implement sigmoid function and its gradient function:
    sigmoid(x) = 1 / (1 + exp(-x))
    if puts x -> simoid(x), the gradient of simoid(x) would be
    sigmoid(x) * (1 - sigmoid(x)).

    :param t:
    :return: Tensor(data,
           requires_grad,
           depends_on)
    """
    t = _ensure_tensor(t)

    return _sigmoid(t = t)


def ReLU(t: 'Tensorable') -> 'Tensor':
    """
    Implement ReLU function and its gradient function:
    ReLU(x) = max(0, t)
    if puts x -> ReLU(x), the gradient of ReLU(x) would be:
    1. 1 for t >= 0;
    2. 0 for t < 0;

    :param t:
    :return: Tensor(data,
               requires_grad,
               depends_on)
    """
    t = _ensure_tensor(t)

    return _ReLU(t = t)


def Leaky_ReLU(t: 'Tensorable', k: 'float' = 0.01) -> 'Tensor':
    """
    Implement Leaky-ReLU function and its gradient function:
    Leaky-ReLU(x) = max(kt, t)
    if puts x -> Leaky-ReLU(x), the gradient of Leaky-ReLU(x) would be:
    1. 1 for t >= 0;
    2. k for t < 0;

    :param k:
    :param t:
    :return: Tensor(data,
               requires_grad,
               depends_on)
    """
    t = _ensure_tensor(t)

    return _Leak_ReLU(t = t, k = k)


def softplus(t: 'Tensorable') -> 'Tensor':
    t = _ensure_tensor(t)

    return _softplus(t)


def softmax(t: 'Tensorable', axis=None) -> 'Tensor':
    """
    Implement softmax function and its gradient function:
    :param axis:
    :param t:
    :return: Tensor(data,
               requires_grad,
               depends_on)
    """
    t = _ensure_tensor(t)

    return _softmax(t = t, axis = axis)

def maximum(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Implement maxium function and its gradient function:
    Takes t1, t2 and return the the max elements in t1 and t2.
    Give a thought to the broadcastsing operation.

    For example:
    ------------
    |>>> a = Tensor([[1, -1], [2, 0], [3, 4]])
    |>>> b = Tensor([1, 2,])
    |>>> maximum(a, b)
    |>>> [[1, 2], [2, 2], [3, 4]]
    :param isnew: decides whether giving result gradient.
    :param t1:
    :param t2:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _maximum(t1 = t1, t2 = t2, isnew = isnew)


def minimum(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Implement maxium function and its gradient function:
    Takes t1, t2 and return the the min elements in t1 and t2.
    Give a thought to the broadcastsing operation.

    For example:
    ------------
    |>>> a = Tensor([[1, -1], [2, 0], [3, 4]])
    |>>> b = Tensor([1, 2,])
    |>>> minimum(a, b)
    |>>> [[1, -1], [1, 0], [1, 2]]
    :param isnew: decides whether giving result gradient.
    :param t1:
    :param t2:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _minimum(t1 = t1, t2 = t2, isnew = isnew)


def one_hot(t: 'Tensorable', depth, on_value=1, off_value=0, isnew: bool = True):
    """
    Returns a one-hot tensor.
    Codes elements in t a return a one-hot tensor.

    For example:
    ------------
    ```python
    a = [0, 1, 2]
    depth = 3
    one_hot(a, depth)   #output: (3, 3)
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]

    indices = [0, 2, -1, 1]
    depth = 3
    one_hot(indices, depth,
             on_value=5.0)  # output: (3, 3)
    [[5.0, 0.0, 0.0],  # one_hot(0)
     [0.0, 0.0, 5.0],  # one_hot(2)
     [0.0, 0.0, 0.0],  # one_hot(-1)
     [0.0, 5.0, 0.0]]  # one_hot(1)
    """
    t = _ensure_tensor(t)

    return _one_hot(t = t, depth = depth, on_value = on_value, off_value = off_value, isnew = isnew)


def ELU(t: 'Tensorable', alpha=1) -> 'Tensor':
    """
    Here implements the ELU function in deep learning, whose expression is:
    max(0, t) + min(0, a * (exp(t) - 1))

    :param alpha:
    :param t:
    :return:
    """
    t = _ensure_tensor(t)

    return maximum(t, 0, isnew = False) + minimum(0, alpha * (exp(t) - 1), isnew = False)


def ReLUx(t: 'Tensorable', x=8.) -> 'Tensor':
    """
    Here implements the ReLUx func for tensor whose expression is:
    min(max(0, t), x)
    :param t:
    :param x:
    :return:
    """
    t = _ensure_tensor(t)

    return minimum(maximum(0, t, isnew = False), x, isnew = False)


def abs(t: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Here implements the abs(x) function for tensor.
    :param t:
    :param isnew:
    :return:
    """
    t = _ensure_tensor(t)

    return _abs(t = t, isnew = isnew)

def where() -> 'Tensor':
    """

    :return:
    """
    raise NotImplementedError
