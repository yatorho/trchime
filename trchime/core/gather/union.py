"""
Basic List been set up here.
"""
import warnings

from ..tensor import Tensor
from numpy import ndarray
from numpy import array, int16, int32, int64, float16, float32, float64
from typing import Union

from .. import dtype as dp

Arrayable = Union[ndarray,
                  Tensor,
                  int,
                  list,
                  float,
                  bool]

Tensorable = Union[float,
                   list,
                   ndarray,
                   int,
                   Tensor,
                   bool]

Intable = Union[int,
                dp.int8,
                dp.int16,
                dp.int32,
                dp.int64,
                float,
                dp.float16,
                dp.float32,
                dp.float64,
                bool]

Floatable = Union[int,
                  dp.int8,
                  dp.int16,
                  dp.int32,
                  dp.int64,
                  float,
                  dp.float16,
                  dp.float32,
                  dp.float64,
                  bool]


def _ensure_array(arrayable: 'Arrayable') -> ndarray or 'None':
    """
    Ensure inputs be a `numpy.ndarray`
    :param arrayable:
    :return:
    """
    if isinstance(arrayable, ndarray):
        return arrayable
    elif isinstance(arrayable, Tensor):
        return array(arrayable.data)
    elif isinstance(arrayable, int) \
            or isinstance(arrayable, float) \
            or isinstance(arrayable, list) \
            or isinstance(arrayable, bool) \
            or isinstance(arrayable, int16) \
            or isinstance(arrayable, int32) \
            or isinstance(arrayable, int64) \
            or isinstance(arrayable, float16) \
            or isinstance(arrayable, float32) \
            or isinstance(arrayable, float64):
        return array(arrayable)

    else:
        warnings.warn(f"failed to cast type {type(arrayable)} into `Array`", FutureWarning)
        return array(arrayable)


def _ensure_tensor(tensorable: 'Tensorable') -> 'Tensor':
    """
    Ensure inputs be a `trchime.Tensor`
    :param tensorable:
    :return:
    """
    if isinstance(tensorable, Tensor):
        return tensorable

    elif isinstance(tensorable, ndarray):

        return Tensor(tensorable)
    elif isinstance(tensorable, int) \
            or isinstance(tensorable, float) \
            or isinstance(tensorable, list) \
            or isinstance(tensorable, bool):
        return Tensor(_ensure_array(tensorable))
    else:
        warnings.warn(f"could't cast type{type(tensorable)} into `Tensor`", FutureWarning)
        return Tensor(_ensure_array(tensorable))
