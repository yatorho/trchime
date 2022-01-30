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
    Here implements the tanh(hyperbolic tangent) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of tanh(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.tanh(a)
    >>> b
    Tensor(
    [[0.76159416 0.96402758]
     [0.99505475 0.9993293 ]], requires_grad=True, dtype=float64)
    >>> x = tce.tanh(3)
    >>> x
    Tensor(
    0.9950547536867305, requires_grad=False, dtype=float64)
    >>> y = tce.trange(6, 9)
    >>> print(tce.tanh(y))
    Tensor(
    [0.99998771 0.99999834 0.99999977], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _tanh(t=t)


def sin(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the sin(sine) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of sin(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.sin(a)
    >>> b
    Tensor(
    [[ 0.84147098  0.90929743]
     [ 0.14112001 -0.7568025 ]], requires_grad=True, dtype=float64)

    >>> x = tce.tanh(1.57)
    >>> x
    Tensor(
    0.9999996829318346, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.tanh(y))
    Tensor(
    [-0.2794155   0.6569866   0.98935825], requires_grad=False, dtype=float64)

    """

    t = _ensure_tensor(t)

    return _sin(t)


def cos(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the cos(cosine) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of cos(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.cos(a)
    >>> b
    Tensor(
    [[ 0.54030231 -0.41614684]
     [-0.9899925  -0.65364362]], requires_grad=True, dtype=float64)

    >>> x = tce.cos(3.14)
    >>> x
    Tensor(
    -0.9999987317275395, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.cos(y))
    Tensor(
    [ 0.96017029  0.75390225 -0.14550003], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _cos(t=t)


def tan(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the tan(tangent) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of tan(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.tan(a)
    >>> b
    Tensor(
    [[ 1.55740772 -2.18503986]
     [-0.14254654  1.15782128]], requires_grad=True, dtype=float64)

    >>> x = tce.tan(1.57)
    >>> x
    Tensor(
    1255.7655915007897, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.tan(y))
    Tensor(
    [-0.29100619  0.87144798 -6.79971146], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _tan(t=t)


def exp(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the exp(exponential) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of exp(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.exp(a)
    >>> b
    Tensor(
    [[ 2.71828183  7.3890561 ]
     [20.08553692 54.59815003]], requires_grad=True, dtype=float64)

    >>> x = tce.exp(tce.log(2))
    >>> x
    Tensor(
    2.0, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.exp(y))
    Tensor(
    [ 403.42879349 1096.63315843 2980.95798704], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _exp(t=t)


def log(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the log(logarithmic) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of log(t)

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.log(a)
    >>> b
    Tensor(
    [[0.         0.69314718]
     [1.09861229 1.38629436]], requires_grad=True, dtype=float64)

    >>> x = tce.log(tce.exp(6))
    >>> x
    Tensor(
    6.0, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.log(y))
    Tensor(
    [1.79175947 1.94591015 2.07944154], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _log(t=t)


def sigmoid(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the sigmoid function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of sigmoid(t) = 1 / (1 + exp(-t))

    Examples:
    >>> a = tce.Tensor([[1, 2],
                        [3, 4]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.sigmoid(a)
    >>> b
    Tensor(
    [[0.73105858 0.88079708]
     [0.95257413 0.98201379]], requires_grad=True, dtype=float64)

    >>> x = tce.sigmoid(tce.log(6))
    >>> x
    Tensor(
    0.8571428571428571, requires_grad=False, dtype=float64)

    >>> y = tce.trange(6, 9)
    >>> print(tce.sigmoid(y))
    Tensor(
    [0.99752738 0.99908895 0.99966465], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _sigmoid(t=t)


def ReLU(t: 'Tensorable') -> 'Tensor':
    """
    Here implements the ReLU(Rectified Linear Units) function for tensors.

    Parameter: t: 'Tensorable'
    Return: Tensor: value of ReLU(x) = max(0, t)

    Examples:
    >>> a = tce.Tensor([[-1, -2],
                        [1, 2]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.ReLU(a)
    >>> b
    Tensor(
    [[0 0]
     [1 2]], requires_grad=True, dtype=int32)

    >>> x = tce.ReLU(0)
    >>> x
    Tensor(
    0, requires_grad=False, dtype=int32)

    >>> y =  tce.log(2)
    >>> print(tce.ReLU(y))
    Tensor(
    0.6931471805599453, requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _ReLU(t=t)


def Leaky_ReLU(t: 'Tensorable', k: 'float' = 0.01) -> 'Tensor':
    """
    Here implements the Leaky_ReLU(Leaky Rectified Linear Units) function for tensors.

    Parameter: t: 'Tensorable'
               k: 'float'
    Return: Tensor: value of Leaky-ReLU(t, k) = max(kt, t)

    Examples:
    >>> a = tce.Tensor([[-1, -3],
                        [1, 3]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Leaky_ReLU(a,0.5)  # t = a, k = 0.5
    >>> b
    Tensor(
    [[-0.5 -1.5 ]  # -1*k = -0.5 > -1 , -3*k = -1.5 > -3
     [ 1.   3. ]], requires_grad=True, dtype=float64)  # 1*k = 0.5 < 1 , 3*0.5 = 1.5 < 3

    >>> x = tce.Leaky_ReLU(tce.Tensor([1, -1, 0]), -0.5)
    >>> x
    Tensor(
    [ 1.   0.5 -0. ], requires_grad=False, dtype=float64)

    >>> y = tce.log(2)
    >>> print(tce.Leaky_ReLU(y, 30))
    Tensor(
    20.79441541679836, requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return _Leak_ReLU(t=t, k=k)


def softplus(t: 'Tensorable') -> 'Tensor':
    t = _ensure_tensor(t)

    return _softplus(t)


def softmax(t: 'Tensorable', axis=None) -> 'Tensor':
    """
    Here implements the softmax function for tensors.

    Parameter: t: 'Tensorable'
               axis: 'Integer': Define the axis of t
    Return: Tensor: value of softmax(t, axis = a)

    Examples:
    >>> a = tce.Tensor([[-1, -3],
                        [1, 3]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.softmax(a)
    >>> b
    Tensor(
    [[0.0158422  0.00214401]
     [0.11705891 0.86495488]], requires_grad=True, dtype=float64)
    # tce.exp(-1) = 0.368
    # tce.exp(-3) = 0.050
    # tce.exp(1) = 2.718
    # tce.exp(3) = 20.086
    # sum = 0.368 + 0.050 + 2.718 + 20.086 = 23.222
    # tce.exp(-1)/sum = 0.0158
    # tce.exp(-3)/sum = 0.0021
    # tce.exp(1)/sum = 0.117
    # tce.exp(3)/sum = 0.864

    >>> a = tce.Tensor([[-1, -2],
                        [1, 3]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.softmax(a, axis = 1)  # axis = 1, the softmax starts from the tensors in 1 axis of a
    >>> b
    Tensor(
    [[0.73105858 0.26894142]
     [0.11920292 0.88079708]], requires_grad=True, dtype=float64)
    # tce.exp(-1) = 0.368
    # tce.exp(-2) = 0.135
    # sum1 = tce.exp(-1) + tce.exp(-2) = 0.368 + 0.135 = 0.503
    # tce.exp(-1)/sum1 = 0.731
    # tce.exp(-3)/sum2 = 0.268

    # tce.exp(1) = 2.718
    # tce.exp(3) = 20.086
    # sum2 = tce.exp(1) + tce.exp(3) = 2.718 + 20.086 = 22.804
    # tce.exp(1)/sum2 = 0.119
    # tce.exp(3)/sum2 = 0.880


    >>> a = tce.Tensor([[-1, -2],
                        [1, 3]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.softmax(a, axis = 0)  # axis = 0, the softmax starts from the tensors in 0 axis of a
    >>> b
    Tensor(
    [[0.11920292 0.00669285]
     [0.88079708 0.99330715]], requires_grad=True, dtype=float64)
    # tce.exp(-1) = 0.368
    # tce.exp(1) = 2.718
    # sum1 = tce.exp(-1) + tce.exp(1) = 0.368 + 2.718 = 3.086
    # tce.exp(-1)/sum1 = 0.119
    # tce.exp(1)/sum2 = 0.880

    # tce.exp(-2) = 0.135
    # tce.exp(3) = 20.086
    # sum2 = tce.exp(-2) + tce.exp(3) = 0.135 + 20.086 = 20.221
    # tce.exp(1)/sum2 = 0.006
    # tce.exp(3)/sum2 = 0.993

    """
    t = _ensure_tensor(t)

    return _softmax(t=t, axis=axis)


def maximum(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Here implements the maximum function for tensors.

    Parameter: t1, t2: 'Tensorable'
               isnew: 'Bool': decides whether giving result gradient
    Return: Tensor: the max elements in t1, t2, value of maximum(t1, t2) = max(t1, t2)
    It's combined with broadcastsing operation.

    Examples:
    >>> a = tce.Tensor([[1,3],
                        [-1,1],
                        [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([2])
    >>> tce.maximum(a, b)
    Tensor(
    [[2 3]
     [2 2]
     [5 6]], requires_grad=False, dtype=int32)

    >>>a = tce.Tensor([[0,3],
                       [-1,1],
                       [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([1,2])
    >>> tce.maximum(a, b)
    Tensor(
    [[1 3]  # 0 < 1, 3 > 2
     [1 2]  # -1 < 1, 1 < 2
     [5 6]], requires_grad=False, dtype=int32)  # 5 > 1, 6 > 2

    >>> a = tce.Tensor([[0,3],
                       [-1,1],
                       [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([[1,2],
                        [-2, 8],
                        [10, 5.5]], requires_grad = True, dtype = tce.float32)
    >>> tce.maximum(a, b)
    Tensor(
    [[ 1.  3.]  # 0 < 1, 3 > 2
     [ -1.  8.]  # -1 > -2, 1 < 8
     [10.  6.]], requires_grad=False, dtype=float64)  # 5 < 10, 6 > 5.5

    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _maximum(t1=t1, t2=t2, isnew=isnew)


def minimum(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Here implements the minimum function for tensors.

    Parameter: t1, t2: 'Tensorable'
               isnew: 'Bool': decides whether giving result gradient
    Return: Tensor: the min elements in t1, t2, value of minimum(t1, t2) = min(t1, t2)
    It's combined with broadcastsing operation.

    Examples:
    >>> a = tce.Tensor([[1,3],
                        [-1,1],
                        [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([2])
    >>> tce.minimum(a, b)
    Tensor(
    [[ 1  2]
     [-1  1]
     [ 2  2]], requires_grad=False, dtype=int32)

    >>>a = tce.Tensor([[0,3],
                       [-1,1],
                       [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([1,2])
    >>> tce.minimum(a, b)
    Tensor(
    [[ 0  2]
     [-1  1]
     [ 1  2]], requires_grad=False, dtype=int32)

    >>> a = tce.Tensor([[0,3],
                       [-1,1],
                       [5,6]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.Tensor([[1,2],
                        [-2, 8],
                        [10, 5.5]], requires_grad = True, dtype = tce.float32)
    >>> tce.minimum(a, b)
    Tensor(
    [[ 0  2]
     [-2  1]
     [ 5  5.5]], requires_grad=False, dtype=float32)

    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _minimum(t1=t1, t2=t2, isnew=isnew)


def one_hot(t: 'Tensorable', depth, on_value=1, off_value=0, isnew: bool = True):
    """
    Here implements the one_hot function for tensors.

    Parameter: t: 'Tensorable'
               depth: optional, 'Integer': the size length of output tensor
               on_value: optional: the number of those on figure
               off_value: optional: the number of those off figure
               param isnew: 'Bool': decides whether giving result gradient
    Return: Tensor: expand t to depth size, It's combined with broadcasting

    Examples:
     >>> a = tce.Tensor([1], requires_grad = True, dtype = tce.int32)
     >>> depth = 3
     >>> tce.one_hot(a, depth)  # output: (1, 3)
     Tensor(
     [[0. 1. 0.]], requires_grad=False, dtype=float64)  # the second on_value is 1, else off_value are 0


     >>> a = tce.Tensor([0, 1, 2], requires_grad = True, dtype = tce.int32)
     >>> depth = 3
     >>> tce.one_hot(a, depth)  # output: (3, 3)
     Tensor(
     [[1. 0. 0.]  # one_hot(0)
      [0. 1. 0.]  # one_hot(1)
      [0. 0. 1.]], requires_grad=False, dtype=float64)  # one_hot(2)

     >>> a = tce.Tensor([0, 2, 2], requires_grad = True, dtype = tce.int32)
     >>> depth = 3
     >>> tce.one_hot(a, depth)  # output: (3, 3)
     Tensor(
     [[1. 0. 0.]  # one_hot(0)
      [0. 0. 1.]  # one_hot(2)
      [0. 0. 1.]], requires_grad=False, dtype=float64)  # one_hot(2)

     >>> a = tce.Tensor([[0, 2, 5],
                        [0, 1, 3]], requires_grad = True, dtype = tce.int32)
     >>> depth = 4
     >>> tce.one_hot(a, depth)  # output: (2, 3, 4)
     Tensor(
     [[[1. 0. 0. 0.]  # one_hot(0)
       [0. 0. 1. 0.]  # one_hot(2)
       [0. 0. 0. 0.]] # one_hot(5)

      [[1. 0. 0. 0.]  # one_hot(0)
       [0. 1. 0. 0.]  # one_hot(1)
       [0. 0. 0. 1.]]], requires_grad=False, dtype=float64)  # one_hot(3)

     >>> a = tce.Tensor([0, 1, 2], requires_grad = True, dtype = tce.int32)
     >>> depth = 3
     >>> tce.one_hot(a, depth, on_value = 6, off_value = 9)  # output: (3, 3)
     Tensor(
     [[6. 9. 9.]  # one_hot(0), on_value = 6, off_value = 9
      [9. 6. 9.]  # one_hot(1), on_value = 6, off_value = 9
      [9. 9. 6.]], requires_grad=False, dtype=float64) # one_hot(2), on_value = 6, off_value = 9

    """
    t = _ensure_tensor(t)

    return _one_hot(t=t, depth=depth, on_value=on_value, off_value=off_value, isnew=isnew)


def ELU(t: 'Tensorable', alpha=1) -> 'Tensor':
    """
    Here implements the ELU(Exponential Linear Unit) function for tensors.

    Parameter: t: 'Tensorable'
               alpha: optional: 'Float'
    Return: Tensor: ELU(t) = t, if t > 0
                           = alpha*(tce.exp(t)-1), if t < 0

    Examples:
    >>> a = tce.Tensor([-1, 1], requires_grad = True, dtype = tce.int32)
    >>> b = tce.ELU(a)
    >>> b
    Tensor(
    [-0.63212056  1.        ], requires_grad=True, dtype=float64)

    >>> a = tce.Tensor([-1, 1], requires_grad = True, dtype = tce.int32)
    >>> b = tce.ELU(a, alpha = 2)
    >>> b
    Tensor(
    [-1.26424112  1.        ], requires_grad=True, dtype=float64)

    """
    t = _ensure_tensor(t)

    return maximum(t, 0, isnew=False) + minimum(0, alpha * (exp(t) - 1), isnew=False)


def ReLUx(t: 'Tensorable', x=8.) -> 'Tensor':
    """
    Here implements the ReLUx function for tensors.

    Parameter: t: 'Tensorable'
               x: 'Float'
    Return: Tensor: value of ReLUx(t) = min(max(0, t), x)

    Examples:
    >>> a = tce.Tensor([[-1, 1],
                        [7, 9]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.ReLUx(a)
    >>> b
    Tensor(
    [[0. 1.]
     [7. 8.]], requires_grad=True, dtype=float64)

    >>> y = tce.trange(6, 10)
    >>> print(tce.ReLUx(y))
    Tensor(
    [6. 7. 8. 8.], requires_grad=False, dtype=float64)

    """
    t = _ensure_tensor(t)

    return minimum(maximum(0, t, isnew=False), x, isnew=False)


def abs(t: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
     Here implements the abs(absolute) function for tensors.

    Parameter: t: 'Tensorable'
               isnew: 'Bool': decides whether giving result gradient
    Return: Tensor: value of abs(t) = t, if t > 0
                                    = -t, if t < 0

    Example:
    >>> a = tce.Tensor([[-1, 1],
                        [-6, 9]], requires_grad = True, dtype = tce.int32)
    >>> b = tce.abs(a)
    >>> b
    Tensor(
    [[1 1]
     [6 9]], requires_grad=False, dtype=int32)

    """
    t = _ensure_tensor(t)

    return _abs(t=t, isnew=isnew)


def where() -> 'Tensor':
    """

    :return:
    """
    raise NotImplementedError
