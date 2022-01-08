"""
Here is moudle for common operations and they gradient for tensor.
"""
from ._multitensor import _sum, _mean, _add, _mul, _pow, _rec, _div, _neg, _sub, _matmul, _dot, _transpose_T, \
    _transpose, _slice, _max, _min, _argmax, _argmin, _eq, _ne, _gt, _lt, _ge, _le, _reshape, _astype, _copy, _inv, \
    _vstack, _stack, _vstack_n, _size, _var, _absolute_equal, _absolute_gt, _absolute_ge, _absolute_lt, _absolute_le, \
    _absolute_negequal, _some_equal, _some_negequal, _some_gt, _some_ge, _some_lt, _some_le
from .tensor import Tensor

from .gather import Tensorable
# noinspection PyProtectedMember
from .gather import _ensure_tensor

from typing import Iterable
import warnings
from typing import Union




def sum(t: 'Tensorable', axis=None, keepdims: bool = False) -> 'Tensor':
    """
    Takes a tensor and returns the tensor
    that's the sum of all its elements.
    :param keepdims:
    :param axis:
    :param t:
    :return: Tensor(data,
                    requires_grad,
                    depends_on)
    """
    t = _ensure_tensor(t)

    return _sum(t = t, axis = axis, keepdims = keepdims)


def var(t: 'Tensorable', axis: int = None, ddof: int = 0, keepdims: bool = False) -> 'Tensor':
    """

    :param t:
    :param axis:
    :param ddof:
    :param keepdims:
    :return:
    """
    t = _ensure_tensor(t)

    return _var(t = t, axis = axis, ddof = ddof, keepdims = keepdims)


def mean(t: 'Tensorable', axis=None, keepdims: bool = False) -> 'Tensor':
    """
    Takes a tensor and returns the tensor
    that's the mean of all its elements.
    :param keepdims:
    :param axis:
    :param t:
    :return: Tensor(data,
                    requires_grad,
                    depends_on)
    """
    # t = ensure_tensor(t)
    # return t.sum(axis = axis, keepdims = keepdims) / np.size(t.data, axis = axis)

    t = _ensure_tensor(t)
    return _mean(t = t, axis = axis, keepdims = keepdims)


def add(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    """
    Here implemention of the add operation for Tensor.
    if t1 + t2 = t3, just:
    t3.data = t1.data + t1.data
    t3.requires_grad is true only command t1's or t2's requires_grad
    is true.

    t3.dependency is a List[Denpendency], maually append when t1, t2 is
    requires gradient.

    The gradient function:
    if t1 = [1, 2, 3], t2 = [4, 5, 6], then
    t3 = t1 + t2 = [5, 7, 9]
    when t1 = [1+e, 2, 3] => t3 = [5+e, 7, 9]
    which means the grad of t1, t2 is just the same as
    t3's grad.

    Especially, give a thought to the tensor's broadcasting operation.

    What's broadcasting:
    1. t1,2 has different ndims:
    1). t1 = 1, t2= [1, 2, 3] => t3 = t1 + t2 = [2, 3, 4]
    2). t1 = [1, 2, 3], t2 = [[1, 2, 3], [4, 5, 6]]
        => t3 = t1 + t2 = [[2, 4, 6], [5, 6, 9]]
    2. t1,2 has same ndims:
    1). t1 = [1], t2 = [1, 2, 3] => t3 = t1 + t2 = [2, 3, 4]
    2). t2 = [[1], [2]], t3 = [[1, 2, 3], [4, 5, 6]]
        => t3 = t1 + t2 = [[2, 3, 4], [6, 7, 8]]

    When first case happened, the grad of t1[i] should be the
    sum of t3's grad[i].
    When second case happened, the grad of t1[i] should be
    sum of t3.grad[i] where length of ith dims of t1 should be
    signle.

    :param t1: addend tensor1
    :param t2: addend tensor2
    :return: Tensor(data,
                  requires_grad,
                  depends_on)
    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _add(t1 = t1, t2 = t2)


def mul(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    """
    Here implemention of tensor's multiplication.
    :param t1:
    :param t2:
    :return:
    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _mul(t1 = t1, t2 = t2)


def pow(t: 'Tensorable', power: 'Tensorable') -> 'Tensor':

    t = _ensure_tensor(t)
    power = _ensure_tensor(power)
    if power.requires_grad:
        warnings.warn('the power tensor assigned with gradient.', FutureWarning)

    return _pow(t = t, power = power)


def rec(t: 'Tensorable') -> 'Tensor':

    t = _ensure_tensor(t)

    return _rec(t = t)


def div(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _div(t1 = t1, t2 = t2)


def neg(t: 'Tensorable') -> 'Tensor':

    t = _ensure_tensor(t)

    return _neg(t = t)


def sub(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _sub(t1 = t1, t2 = t2)


def matmul(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    """
    if t3 = t1@t2, and grad3 is gradient of some function wrt t3, then
    grad1 = grad @ t2.T
    grad2 = t1.T @ grad
    :param t1:
    :param t2:
    :return:Tensor(data,
                   requires_grad,
                   depends_on)
    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _matmul(t1 = t1, t2 = t2)


def dot(t1: 'Tensorable', t2: 'Tensorable') -> 'Tensor':
    """

    :param t1:
    :param t2:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _dot(t1 = t1, t2 = t2)


def transpose_T(t: 'Tensorable') -> 'Tensor':
    """
    here implement the transpose operation for tensor.
    if t = [[1], [2], [3]], then, t.T = [[1, 2, 3]]
    especially, if t = [1, 2, 3], t.T = [1, 2, 3]
    maually, the gradient of t is the transposed tensor of
    t.T.
    :param t:
    :return:Tensor(data,
                   requires_grad,
                   depends_on)
    """

    t = _ensure_tensor(t)

    return _transpose_T(t = t)


def transpose(t: 'Tensorable', *axes) -> 'Tensor':
    """
    a.transpose(*axes)

    Returns a view of the array with axes transposed.

    For a 1-D array this has no effect, as a transposed vector is simply the
    same vector. To convert a 1-D array into a 2D column vector, an additional
    dimension must be added. `np.atleast2d(a).T` achieves this, as does
    `a[:, np.newaxis]`.
    For a 2-D array, this is a standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided and
    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

    * None or no argument: reverses the order of the axes.

    * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
    `i`-th axis becomes `a.transpose()`'s `j`-th axis.

    * `n` ints: same as an n-tuple of the same ints (this form is
    intended simply as a "convenience" alternative to the tuple form)

    Returns
    -------
    out : ndarray
    View of `a`, with axes suitably permuted.

    See Also
    --------
    ndarray.T : Array property returning the array transposed.
    ndarray.reshape : Give a new shape to an array without changing its data.
    Examples:
    >>> import trchime as tce
    >>> a = tce.trange(6).reshape((2, 3))
        Tensor( [[0, 1, 2],
                 [3, 4, 5]], requies_grad = False, dtype = tce.int32)
    >>> a.transpose(1, 0)
    Tensor( [[0, 3],
             [1, 4]
             [2, 5]], requies_grad = False, dtype = tce.int32)

    :param t:
    """


    t = _ensure_tensor(t)

    return _transpose(t, *axes)


def slice(t: 'Tensorable', idxs: slice or tuple, isnew: bool = True) -> 'Tensor':
    """
    key to implement the slice operation.

    :param isnew:
    :param t:
    :param idxs:
    :return:
    """

    t = _ensure_tensor(t)

    return _slice(t = t, idxs = idxs, isnew = isnew)


def max(t: 'Tensorable', axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
    """
    key to implement the max operation.
    :param isnew:
    :param keepdims:
    :param t:
    :param axis:
    :return:
    """


    t = _ensure_tensor(t)

    return _max(t = t, axis = axis, keepdims = keepdims, isnew = isnew)


def min(t: 'Tensorable', axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
    """
    key to implement the min operation.
    :param isnew:
    :param keepdims:
    :param t:
    :param axis:
    :return:
    """


    t = _ensure_tensor(t)

    return _min(t = t, axis = axis, keepdims = keepdims, isnew = isnew)


def argmax(t: 'Tensorable', axis=None, isnew: bool = True) -> 'Tensor':
    """
    key to implement the argmax operation.
    Return arg of the max element in tensor, which means
    the argmax(x) is a discontinuouos function. So, the grad_fn
    just return a zero tensor whose shape likes t.

    :param isnew:
    :param t:
    :param axis:
    :return:
    """

    t = _ensure_tensor(t)

    return _argmax(t = t, axis = axis, isnew = isnew)


def argmin(t: 'Tensorable', axis=None, isnew: bool = True) -> 'Tensor':
    """
    key to implement the argmin operation.
    Return arg of the min element in tensor, which is also
    a 'Tensor" with not-requires_grad.

    :param isnew:
    :param t:
    :param axis:
    :return:
    """


    t = _ensure_tensor(t)

    return _argmin(t = t, axis = axis, isnew = isnew)


def eq(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Takes two tensors t1, t2, and gives the t3 := t1 == t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The eq function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _eq(t1 = t1, t2 = t2, isnew = isnew)


def ne(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Takes two tensors t1, t2, and gives the t3 := t1 != t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The ne function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """


    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _ne(t1 = t1, t2 = t2, isnew = isnew)


def gt(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    greater than

    Takes two tensors t1, t2, and gives the t3 := t1 > t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The gt function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """


    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _gt(t1 = t1, t2 = t2, isnew = isnew)


def lt(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    lower than

    Takes two tensors t1, t2, and gives the t3 := t1 < t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The lt function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """


    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _lt(t1 = t1, t2 = t2, isnew = isnew)


def ge(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    greater equal

    Takes two tensors t1, t2, and gives the t3 := t1 >= t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The ge function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _ge(t1 = t1, t2 = t2, isnew = isnew)


def le(t1: 'Tensorable', t2: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    lower equal
    Takes two tensors t1, t2, and gives the t3 := t1 <= t2.
    t3.data would be assigned with type: bool.
    You can use operation: 1*t2 to change its value to type: int.
    The le function' result would be discountinuous, so the gradient
    func just return a zero tensor whose shape likes t1 or t2.
    :param isnew:
    :param t1:
    :param t2:
    :return:
    """


    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _le(t1 = t1, t2 = t2, isnew = isnew)


def absolute_equal(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param only_value:
    :param t1:
    :param t2:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_equal` key command same shapes')

    return _absolute_equal(t1 = t1, t2 = t2, only_value = only_value)

def absolute_negequal(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param only_value:
    :param t1:
    :param t2:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_negequal` key command same shapes')

    return _absolute_negequal(t1 = t1, t2 = t2, only_value = only_value)


def absolute_gt(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_gt` key command same shapes')

    return _absolute_gt(t1 = t1, t2 = t2, only_value = only_value)


def absolute_ge(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_ge` key command same shapes')

    return _absolute_ge(t1 = t1, t2 = t2, only_value = only_value)


def absolute_lt(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_lt` key command same shapes')

    return _absolute_lt(t1 = t1, t2 = t2, only_value = only_value)


def absolute_le(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _absolute_le(t1 = t1, t2 = t2, only_value = only_value)


def some_equal(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_equal(t1 = t1, t2 = t2, only_value = only_value)

def some_negequal(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_negequal(t1 = t1, t2 = t2, only_value = only_value)

def some_gt(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_gt(t1 = t1, t2 = t2, only_value = only_value)

def some_ge(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_ge(t1 = t1, t2 = t2, only_value = only_value)


def some_lt(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> bool:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_lt(t1 = t1, t2 = t2, only_value = only_value)

def some_le(t1: 'Tensorable', t2: 'Tensorable', only_value: bool = True) -> True:
    """

    :param t1:
    :param t2:
    :param only_value:
    :return:
    """
    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    if t1.shape != t2.shape:
        raise ValueError('`absolute_le` key command same shapes')

    return _some_le(t1 = t1, t2 = t2, only_value = only_value)



def reshape(t: 'Tensorable', shape) -> 'Tensor':
    """


    Gives a new shape to a tensor without changing its elements.
    :param t:
    :param shape:
    :return:
    """


    t = _ensure_tensor(t)

    return _reshape(t = t, shape = shape)


def astype(t: Tensorable, dtype: Union[str, object]) -> 'Tensor':
    """
    Casting the elements's type in tensor.
    :param t:
    :param dtype:
    :return:
    """
    t = _ensure_tensor(t)

    return _astype(t = t, dtype = dtype)


def copy(t: 'Tensorable', only_data: bool = True):
    """
    Takes a tensor, returns a new tensor with same data, different address.

    Parameters
    -----------
    t: tensor_like,
       Input Data
    only_data: whether copy t's grad
    :param t:
    :param only_data:
    :return:
    """


    t = _ensure_tensor(t)

    return _copy(t = t, only_data = only_data)


def inv(t: 'Tensorable', isnew: bool = True) -> 'Tensor':
    """
    Implements the inverse operation for matrix.

    :param isnew:
    :param t:
    :return:
    """



    t = _ensure_tensor(t)

    if not isnew and t.requires_grad:
        warnings.warn('called non-realized gradient for inverse operation', RuntimeWarning)

    return _inv(t = t, isnew = isnew)


def square(t: 'Tensorable') -> 'Tensor':
    """
    Takes a tensor and return it's square value.
    :param t:
    :return:
    """
    return pow(t, 2)


def sqrt(t: 'Tensorable') -> 'Tensor':
    """
    Takes a tensor and return it's sqrt value.
    :param t:
    :return:
    """
    return pow(t, 0.5)


def vstack(t1: 'Tensorable', t2: 'Tensorable', isnew=True) -> 'Tensor':
    """
    Stack tensors in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    t1 : array_like or tensor_like
    t2 : arrar_like or tnesor_like
    isnew  : bool, optional
    The tensors must have the same shape along all but the first axis.
    1-D tesnros must have the same length.

    Returns
    -------
    stacked : tesnor
    The array formed by stacking the given arrays, will be at least 2-D.

    Examples
    --------
    >>> import trchime as tce
    >>> a = tce.Tensor([1, 2, 3])
    >>> b = tce.Tensor([2, 3, 4])
    >>> tce.vstack(a, b)
    tensor([[1, 2, 3],
            [2, 3, 4]])

    >>> a = tce.Tensor([[1], [2], [3]])
    >>> b = tce.Tensor([[2], [3], [4]])
    >>> tce.vstack(a, b)
    tensor([[1],
            [2],
            [3],
            [2],
            [3],
            [4]])

    """

    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _vstack(t1 = t1, t2 = t2, isnew = isnew)


def vstack_n(tup: Iterable[Tensorable]) -> 'Tensor':
    for i, t in enumerate(tup):
        tup[i] = _ensure_tensor(t)

    return _vstack_n(tup = tup)


def stack(t1: 'Tensorable', t2: 'Tensorable', axis: int = 0,
          isnew: bool = False) -> 'Tensor':
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    .. versionadded:: 1.10.0

    Parameters
    ----------
    t1 : sequence of array_like
    t2 : sequence of array_like
    Each array must have the same shape.

    axis : int, optional
    The axis in the result array along which the input arrays are stacked.

    isnew: bool, optional

    Returns
    -------
    stacked : ndarray
    The stacked array has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.

    Examples
    --------
    >>> import trchime as tce
    >>> t1 = tce.random.randn(3, 4)
    >>> t2 = tce.random.randn(3, 4)
    >>> tce.stack(t1, t2).shape
    (2, 3, 4)

    >>> tce.stack(t1, t2, axis=1).shape
    (3, 2, 4)

    >>> tce.stack(t1, t2, axis=2).shape
    (3, 4, 2)

    >>> a = tce.Tensor([1, 2, 3])
    >>> b = tce.Tensor([2, 3, 4])
    >>> tce.stack((a, b))
    tesnor([[1, 2, 3],
            [2, 3, 4]])

    >>> tce.stack((a, b), axis=-1)
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """


    t1 = _ensure_tensor(t1)
    t2 = _ensure_tensor(t2)

    return _stack(t1 = t1, t2 = t2, axis = axis, isnew = isnew)


def size(t: 'Tensorable', axis=None, isnew: bool = True) -> 'Tensor':
    """

    :param isnew:
    :param t:
    :param axis:
    :return:
    """
    t = _ensure_tensor(t)

    return _size(t = t, axis = axis, isnew = isnew)
