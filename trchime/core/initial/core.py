"""
Here implements the main initialzie methods for tensor.
The package would be imported to trchime.
"""

from ..tensor import Tensor
from ..gather import Tensorable
# noinspection PyProtectedMember
from ..gather import _ensure_tensor

import numpy as np

e = np.e
pi = np.pi

euler_gamma = np.euler_gamma


def eye(N: int, M=None, k=0, dtype=None, requires_grad: bool = False) -> 'Tensor':
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.
    Returns
    -------
    I : tensor of shape (N,M)
      An tensor where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    Examples
    --------
    >>> import trchime as tce
    >>> tce.eye(2, dtype=int)
    [[1 0]
     [0 1]]
    >>> tce.eye(3, k=1)
    [[0.,  1.,  0.],
     [0.,  0.,  1.],
     [0.,  0.,  0.]]
    :param N:
    :param M:
    :param k:
    :param dtype:
    :param requires_grad:
    :return:Tensor(data,
                   requires_grad,
                   depends_on)
    """
    data = np.eye(N = N, M = M, k = k, dtype = dtype)
    requires_grad = requires_grad
    depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def zeros(shape, dtype=None, requires_grad: bool = False) -> 'Tensor':
    """
    zeros(shape, dtype)

    Return a new tensor of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
    Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
    Returns
    -------
    out : ndarray
    Array of zeros with the given shape, dtype, and order.

    Examples
    --------
    >>> import trchime as tce
    >>> tce.zeros(5)
    [ 0.,  0.,  0.,  0.,  0.]

    >>> tce.zeros((5,), dtype=int)
    [0, 0, 0, 0, 0]

    >>> tce.zeros((2, 1))
    [[ 0.],
     [ 0.]]

    >>> s = (2,2)
    >>> tce.zeros(s)
    [[ 0.,  0.],
     [ 0.,  0.]]
    :param shape:
    :param dtype:
    :param requires_grad:
    """
    data = np.zeros(shape = shape, dtype = dtype)
    requires_grad = requires_grad
    depends_on = []
    return Tensor(data,
                  requires_grad,
                  depends_on)


def ones(shape, dtype=None, requires_grad: bool = True) -> 'Tensor':
    """
    Return a new tensor of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    requires_grad : bool, optional
    If True, returned tensor would be assigned with need-gradient.

    Returns
    -------
    out : tensor
        Tensor of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return a tensor of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.


    Examples
    --------
    >>> import trchime as tce
    >>> tce.ones(5)
    tensor([1., 1., 1., 1., 1.])

    >>> tce.ones((5,), dtype=int)
    tensor([1, 1, 1, 1, 1])

    >>> tce.ones((2, 1))
    tensor([[1.],
            [1.]])

    >>> s = (2,2)
    >>> tce.ones(s)
    tensor([[1.,  1.],
            [1.,  1.]])

    """

    data = np.ones(shape = shape, dtype = dtype)
    requires_grad = requires_grad
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def ones_like(t: Tensorable, dtype=None, requires_grad: bool = False) -> 'Tensor':
    """
   Return a tensor of ones with the same shape and type as a given array.

   Parameters
   ----------
   t : tensor_like
       The shape and data-type of `a` define these same attributes of
       the returned array.
   dtype : data-type, optional
       Overrides the data type of the result.

       .. versionadded:: 1.6.0
   requires_grad : bool, optional
   If True, returned tensor would be assigned with need-gradient.

   Returns
   -------
   out : tensor
       Tensor of ones with the same shape and type as `a`.

   See Also
   --------
   empty_like : Return an empty array with shape and type of input.
   zeros_like : Return an array of zeros with shape and type of input.
   full_like : Return a new array with shape of input filled with value.
   ones : Return a new array setting values to one.

   Examples
   --------
   >>> import trchime as tce
   >>> x = tce.arange(6)
   >>> x = x.reshape((2, 3))
   >>> x
   tensor([[0, 1, 2],
          [3, 4, 5]])
   >>> tce.ones_like(x)
   tensor([[1, 1, 1],
          [1, 1, 1]])

   >>> y = tce.arange(3, dtype=float)
   >>> y
   tensor([0., 1., 2.])
   >>> tce.ones_like(y)
   tensor([1.,  1.,  1.])

   """
    t = _ensure_tensor(t)

    return zeros(t.shape, dtype = dtype, requires_grad = requires_grad)


def zeros_like(t: Tensorable, dtype=None, requires_grad: bool = False) -> 'Tensor':
    """
    Return a tensor of zeros with the same shape and type as a given array.

    Parameters
    ----------
    t : tensor_like
        The shape and data-type of `t` define these same attributes of
        the returned tensor.
    dtype : data-type, optional
        Overrides the data type of the result.


    :param t: tensor_like
    :param dtype:  data-type
    :param requires_grad: whether the tensor is requires gradient.
    :return: Tensor of zeros with same shape and type as `t`
    """

    t = _ensure_tensor(t)
    return zeros(t.shape, dtype = dtype, requires_grad = requires_grad)


def arange(start=None, *args, **kwargs) -> 'Tensor':
    """
    arange([start,] stop[, step,], dtype=None, requires_grad=False)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.

    Parameters:
    ----------
    start : number, optional
    Start of interval.  The interval includes this value.  The default
    start value is 0.
    stop : number
    End of interval.  The interval does not include this value, except
    in some cases where `step` is not an integer and floating point
    round-off affects the length of `out`.
    step : number, optional
    Spacing between values.  For any output `out`, this is the distance
    between two adjacent values, ``out[i+1] - out[i]``.  The default
    step size is 1.  If `step` is specified as a position argument,
    `start` must also be given.
    dtype : dtype
    The type of the output array.  If `dtype` is not given, infer the data
    type from the other input arguments.
    requires_grad: bool
    Whether tensor would be assigned with need-grad. The default value of
    `requires_grad` is False.

    Returns
    -------
    Tensor : tensor

    Examples
    --------
    >>> import trchime as tce
    >>> tce.arange(3)
    tensor([0, 1, 2], requires_grad = False)
    >>> tce.arange(3.0, requires_grad = True)
    tensor([ 0.,  1.,  2.], requires_grad = True)
    >>> tce.arange(3,7, dtype = tce.float32)  # data's type is tce.float32
    tensor([3, 4, 5, 6], requires_grad = False)
    >>> tce.arange(3,7,2)
    tensor([3, 5], requires_grad = False)
    """

    if len(args) == 0:
        data = np.arange(start, dtype = kwargs.get('dtype'))
    elif len(args) == 1:
        data = np.arange(start, args[0], dtype = kwargs.get('dtype'))
    elif len(args) == 2:
        data = np.arange(start, args[0], args[1], dtype = kwargs.get('dtype'))
    else:
        raise ValueError('given exceed argments')

    if kwargs.get('requires_grad', False):
        requires_grad = True
    else:
        requires_grad = False

    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def trange(start=None, *args, **kwargs) -> 'Tensor':
    """
    See:
    -------
    arange

    :return:
    """
    return arange(start, *args, **kwargs)


def linspace(start: Tensorable,
             stop: Tensorable,
             num: int = 50,
             endpoint=True,
             retstep=False,
             dtype=None,
             axis: int = 0,
             requires_grad: bool = False) -> 'Tensor' or float:
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : tensor_like
        The starting value of the sequence.
    stop : tensor_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.
    requires_grad: bool, optional
        If True, return a tensor with need-grad.


    Returns
    -------
    samples : tensor
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).

    Examples
    --------
    >>> import trchime as tce
    >>> tce.linspace(2.0, 3.0, num=5)
    tensor([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> tce.linspace(2.0, 3.0, num=5, endpoint=False)
    tensor([2. ,  2.2,  2.4,  2.6,  2.8])
    >>> tce.linspace(2.0, 3.0, num=5, retstep=True)
    (tensor([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
    """

    start = _ensure_tensor(start)
    stop = _ensure_tensor(stop)

    data, step = np.linspace(start = start.data,
                             stop = stop.data,
                             num = num,
                             endpoint = endpoint,
                             retstep = True,
                             dtype = dtype,
                             axis = axis)

    requires_grad = requires_grad

    depends_on = []

    if retstep:
        return Tensor(data,
                      requires_grad,
                      depends_on), step
    else:
        return Tensor(data,
                      requires_grad,
                      depends_on)
