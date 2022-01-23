"""
The tensor moudle.
Here is core part of core library.
Basic operation data in liabray is called "tensor", which
is implemented by class "Tensor".
'tensor' is improved on the basis of the 'numpy' package, and
supports the operation of 'np.ndarray'.
More, 'Tensor' reliazed the gradient function for the array and
it's operation.

"""

from typing import List, Optional, Union, Iterator
import numpy as np

# from .dependency import Dependency
from .dtype import float64


class tensor(object):
    """
    The basic dtype `tensor` in trchime.
    """


class Tensor(tensor):
    """
    Basic data for automatic differential.

    """
    compute_tree_depth = 0

    def __init__(self,
                 data: 'Arrayable',
                 requires_grad: 'bool' = False,
                 depends_on: 'List[Dependency]' = None,
                 dtype: Union[str, object] = None) -> None:

        if dtype is None:
            self._data = _ensure_array(data)
            self._dtype = self._data.dtype
        else:
            self._data = _ensure_array(data).astype(dtype = dtype)
            self._dtype = dtype
        self._requires_grad = requires_grad
        self._depends_on = depends_on or []

        self._shape = self._data.shape
        self._ndim = self._data.ndim
        self._size = self._data.size

        self._grad: Optional['Tensor'] = None

        if self._requires_grad:
            self.zero_grad()

        super().__init__()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> 'None':
        self._data = new_data
        # Setting the data maually means invalidate the gradient.
        if self.requires_grad:
            self.zero_grad()

    @property
    def grad(self) -> 'Tensor':

        return self._grad

    @property
    def depends_on(self) -> List['Dependency']:
        return self._depends_on

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def dtype(self) -> object:
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def T(self) -> 'Tensor':
        return transpose_T(self)

    def zero_grad(self) -> None:
        self._grad = Tensor(np.zeros_like(self.data, dtype = float64))

    def __repr__(self) -> str:
        return f'Tensor(\n{self.data}, requires_grad={self.requires_grad}, dtype={self.dtype})'

    def __str__(self) -> str:
        return self.__repr__()

    def __abs__(self, isnew: bool = True):
        return abs(self, isnew = isnew)

    def __add__(self, other) -> 'Tensor':
        """" gets called for operation: self + other"""
        return add(self, other)

    def __radd__(self, other) -> 'Tensor':
        """ gets called for operation: other + self"""
        return add(other, self)

    def __iadd__(self, other) -> 'Tensor':
        """ gets called for operation: self += ohter"""
        return add(self, other)

    def __mul__(self, other) -> 'Tensor':
        """" gets called for operation: self * other"""
        return mul(self, other)

    def __rmul__(self, other) -> 'Tensor':
        """ gets called for operation: other * self"""
        return mul(other, self)

    def __imul__(self, other) -> 'Tensor':
        """ gets called for operation: self *= other"""
        return mul(self, other)

    def __truediv__(self, other) -> 'Tensor':
        """ gets called for opeartion: self / other"""
        return div(self, other)

    def __rtruediv__(self, other):
        """ gets called for operation: other / self"""
        return div(other, self)

    def __itruediv__(self, other) -> 'Tensor':
        """" gest called for operation: self /= oher"""
        return div(self, other)

    def __matmul__(self, other) -> 'Tensor':
        """ gets called for operation: self @ other"""
        return matmul(self, other)

    def __rmatmul__(self, other):
        """ gets called for operation: other @ self"""
        return matmul(other, self)

    def __neg__(self) -> 'Tensor':
        """ gets called for opertion: -self"""
        return neg(self)

    def __sub__(self, other) -> 'Tensor':
        """ gets called for operation: self - other"""
        return sub(self, other)

    def __rsub__(self, other) -> 'Tensor':
        """ gets called for operation: other - self"""
        return sub(other, self)

    def __isub__(self, other) -> 'Tensor':
        """ gets called for operation: self -= oher"""
        return sub(self, other)

    def __pow__(self, power, modulo=None) -> 'Tensor':
        """ gets called for operation: self ** other """
        return pow(self, power)

    def __getitem__(self, idxs) -> 'Tensor':
        return slice(self, idxs)

    def __setitem__(self, key, value):
        self.data[key] = _ensure_array(value)

    def __eq__(self, other) -> 'Tensor':
        """
        gets called for operation: self == other.

        Also see:
        --------
        .operatons.eq
        """
        return eq(self, other)

    def __ne__(self, other) -> 'Tensor':
        """
        gets called for operation: self != other.
        Also see:
        --------
        .operatons.ne
        """
        return ne(self, other)

    def __gt__(self, other) -> 'Tensor':
        """
        gets called for operation: self > other.
        Also see:
        ---------
        .perations.gt
        """
        return gt(self, other)

    def __lt__(self, other) -> 'Tensor':
        """
        gets called for operation: self < other.
        Also see:
        ---------
        .perations.lt
        """
        return lt(self, other)

    def __ge__(self, other) -> 'Tensor':
        """
        gets called for operation: self >= other.
        Also see:
        ---------
        .perations.ge
        """
        return ge(self, other)

    def __le__(self, other) -> 'Tensor':
        """
        gets called for operation: self <= other.
        Also see:
        ---------
        .perations.le
        """
        return le(self, other)

    def __iter__(self) -> 'Iterator':
        """ override iterable for `Tensor`."""
        return iter(self._data)  # return _data's iter

    def __next__(self) -> 'Tensor':
        pass

    def __copy__(self, only_data: bool = True):
        return copy(self, only_data = only_data)

    def astype(self, dtype: Union[str, object]) -> 'Tensor':
        """
        Casting the elements's type of self.
        Also see:
        ---------
        .operations.astype
        """
        return astype(self, dtype = dtype)

    def _nondepends_on(self) -> 'None':
        """ remove self's dependency"""
        self._depends_on = []

    def assign_add(self, val) -> 'None':
        self.data += _ensure_array(val)

    def assign_sub(self, val) -> 'None':
        self.data -= _ensure_array(val)

    def assign_mul(self, val) -> 'None':
        self.data *= _ensure_array(val)

    def assign_div(self, val) -> 'None':
        self.data /= _ensure_array(val)

    def assign_mal(self, val) -> 'None':
        self.data = self.data @ _ensure_array(val)

    def assign_rmal(self, val) -> 'None':
        self.data = _ensure_array(val) @ self

    def sum(self, axis=None, keepdims: bool = False) -> 'Tensor':
        """
        return the sum of elements in tensor.
        Also see:
        ---------
        .operations.sum
        """
        return sum(self, axis = axis, keepdims = keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        """
        return the mean value of elements in tensor.
        Also see:
        ----------
        .operations.mean
        """
        return mean(self, axis = axis, keepdims = keepdims)

    def max(self, axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
        return max(self, axis = axis, keepdims = keepdims, isnew = isnew)

    def min(self, axis=None, keepdims: bool = False, isnew: bool = True) -> 'Tensor':
        return min(self, axis = axis, keepdims = keepdims, isnew = isnew)

    def argmax(self, axis=None, isnew: bool = True) -> 'Tensor':
        return argmax(self, axis = axis, isnew = isnew)

    def argmin(self, axis=None, isnew: bool = True) -> 'Tensor':
        return argmin(self, axis = axis, isnew = isnew)

    def reshape(self, shape) -> 'Tensor':
        return reshape(self, shape)

    def var(self, axis=None, ddof: int = 0, keepdims: bool = False) -> 'Tensor':
        return var(self, axis, ddof, keepdims)

    def absolute_equal(self, other, only_value: bool = True) -> 'bool':
        return absolute_equal(self, other, only_value)

    def absolute_negequal(self, other, only_value: bool = True) -> 'bool':
        return absolute_negequal(self, other, only_value)

    def absolute_gt(self, other, only_value: bool = True) -> bool:
        return absolute_gt(self, other, only_value)

    def absolute_ge(self, other, only_value: bool = True) -> bool:
        return absolute_ge(self, other, only_value)

    def absolute_lt(self, other, only_value: bool = True) -> bool:
        return absolute_lt(self, other, only_value)

    def absolute_le(self, other, only_value: bool = True) -> bool:
        return absolute_le(self, other, only_value)

    def some_equal(self, other, only_value: bool = True) -> bool:
        return some_equal(self, other, only_value)

    def some_negequal(self, other, only_value: bool = True) -> bool:
        return some_negequal(self, other, only_value)

    def some_lt(self, other, only_value: bool = True) -> bool:
        return some_lt(self, other, only_value)

    def some_le(self, other, only_value: bool = True) -> bool:
        return some_le(self, other, only_value)

    def some_gt(self, other, only_value: bool = True) -> bool:
        return some_gt(self, other, only_value)

    def some_ge(self, other, only_value: bool = True) -> bool:
        return some_ge(self, other, only_value)

    def copy(self, only_data: bool = True) -> 'Tensor':
        return copy(self, only_data = only_data)

    def to_array(self) -> 'np.ndarray':
        """
        return data as a new `numpy.ndarray`.
        :return:
        """
        return np.array(self.data)

    def transpose(self, *axes) -> 'Tensor':
        return transpose(self, *axes)

    def non_gradient(self) -> 'None':
        """ Invalidate tensor's requires_grad property"""
        if self.requires_grad:
            self._requires_grad = False
            self._grad = None
            self._depends_on = []

    def non_depends_on(self) -> 'None':
        """ Invalidaate tesnsor's depends_on property"""
        self._depends_on = []


    def backward(self, grad: 'Tensor' = None) -> None:

        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.)  # dy = 1
            else:
                raise ValueError("grad must be specified for non-0-tensor")

        self._grad.data += grad.data

        for dependency in self._depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


from .multitensor import sum, mean
from .multitensor import add
from .multitensor import sub
from .multitensor import mul
from .multitensor import neg
from .multitensor import matmul
from .multitensor import slice
from .multitensor import div
from .multitensor import pow
from .multitensor import max
from .multitensor import min
from .multitensor import transpose_T
from .multitensor import transpose
from .multitensor import argmax, argmin
from .multitensor import eq, ne, gt, lt, ge, le
from .multitensor import reshape
from .multitensor import astype
from .multitensor import copy
from .multitensor import var
from .multitensor import absolute_equal, absolute_negequal, absolute_gt, absolute_ge, absolute_le, absolute_lt
from .multitensor import some_equal, some_negequal, some_gt, some_ge, some_lt, some_le

from .func import abs

# noinspection PyProtectedMember
from .gather.union import _ensure_array
from .gather import Arrayable
from .dependency import Dependency
