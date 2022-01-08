"""
The function module in for nerual networks.
"""
import warnings
from typing import Iterable, Tuple, Union

from ._core import _conv_tensor2D, _maxpooling_tensor2D, _dropout_layer_tensor, _meanpooling_tensor4D
from ..core.gather import Tensorable
# noinspection PyProtectedMember
from ..core.gather import _ensure_tensor, Arrayable, _ensure_array
from ..core.tensor import Tensor

import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Figure out what the size of the output should be

    """
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    out_height = int(out_height)
    out_width = int(out_width)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def image_to_column(data: np.ndarray, filter_h: int, filter_w: int,
                    stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    assert that data.shape is (batch_num, channel, height, width)
    @params data: 4D input array
    @params filter_h, filter_w: the height and width of the filter
    @returns: the column of input array
    """
    N, C, H, W = data.shape
    assert (H + 2 * padding - filter_h) % stride == 0
    assert (W + 2 * padding - filter_w) % stride == 0
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    im = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    i0 = np.tile(np.repeat(np.arange(filter_h), filter_w), C)
    i1 = np.repeat(np.arange(out_h), out_w) * stride
    j0 = np.tile(np.arange(filter_w), filter_h * C)
    j1 = np.tile(np.arange(out_w), out_h) * stride

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), filter_h * filter_w).reshape(-1, 1)

    col = im[:, k, i, j]
    col = col.transpose(1, 2, 0).reshape(filter_h * filter_w * C, -1)
    return col


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 0, 2).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    An implementation of col2im based on fancy indexing and np.add.at
    """

    warnings.warn("`col2im_indices`has been deprated", DeprecationWarning)

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype = cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def convolution(img: 'Arrayable', filter: 'Arrayable', bias: 'Arrayable', pad: int = 0, stride: int = 1):
    warnings.warn("`convolution` method has been deprecated. take `convolution_layer` instead", DeprecationWarning)

    img = _ensure_array(img)
    filter = _ensure_array(filter)
    bias = _ensure_array(bias)

    N, C, H, W = img.shape
    fN, fC, fH, fW = filter.shape

    out_h = int(1 + (H + 2 * pad - fH) / stride)
    out_w = int(1 + (W + 2 * pad - fW) / stride)

    col = im2col_indices(img, fH, fW, pad, stride).T
    col_w = filter.reshape((fN, -1)).T
    col_b = bias.reshape((fN, 1)).T

    out = np.dot(col, col_w) + col_b
    out = out.reshape((N, out_h, out_w, -1)).transpose(0, 3, 1, 2)  # change ndims for result.

    return out


def convolution_layer2D(img: 'Tensorable', filter: 'Tensorable', bias: 'Tensorable', pad: int = 0,
                        stride: int = 1) -> 'Tensor':
    """
    Also see
    ---------
    trchime.nn._core._conv_tensor4D

    :param img:
    :param filter:
    :param bias:
    :param pad:
    :param stride:
    :return:
    """

    img = _ensure_tensor(img)
    filter = _ensure_tensor(filter)
    bias = _ensure_tensor(bias)

    if img.ndim != 4:
        raise ValueError(f"img's ndim failed matching. use convolution_layer{img.ndim}D instead")
    elif filter.ndim != 4:
        raise ValueError(f"filter's ndim failed matching. use {filter.ndim} convolution instead")
    elif bias.ndim != 2:
        raise ValueError(f"bias's ndim failed matching. use other coonvolutionor instead")

    N, C, H, W = img.shape
    fN, fC, fH, fW = filter.shape

    if not isinstance(pad, int):
        raise TypeError(f"given unexpected type value: {type(pad)} to padding")
    if not isinstance(stride, int):
        raise TypeError(f"given unexpected type value: {type(stride)} to stride")

    if fH - 1 - pad < 0 or fW - 1 - pad < 0:
        raise warnings.warn(
            f"filters would take no effect on img's edge for filter's shape:({fH}, {fW}); padding: {pad}",
            FutureWarning)

    elif not C == fC:
        raise ValueError("filter's chanels' nums failed matching inputs.")
    elif (H + 2 * pad - fH) % stride != 0 or (W + 2 * pad - fW) % stride != 0:
        raise ValueError("dimensions error in convolution's operation")
    elif bias.shape[0] != fN:
        raise ValueError("bias's nums failed mathcing with filters")

    return _conv_tensor2D(img, filter, bias, pad, stride)


def maxpooling_layer2D(img: 'Tensorable', f_shape: Union[Iterable, Tuple[int]] = (2, 2), pad: int = 0,
                       stride: int = 2) -> 'Tensor':
    """


    :param img:
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    img = _ensure_tensor(img)

    if (not isinstance(f_shape, Iterable)) and (not isinstance(f_shape, Tuple)):
        raise ValueError(f'could not cast {type(f_shape)} to tuple.')
    if (len(f_shape)) != 2:
        raise ValueError(f'filter\' shape given unexpected length')

    N, C, H, W = img.shape
    fH, fW = f_shape

    if (H + 2 * pad - fH) % stride != 0 or (W + 2 * pad - fW) % stride != 0:
        raise ValueError('dimensions error in maxpooling layer')

    return _maxpooling_tensor2D(img, (fH, fW), pad, stride)


def meanpooling_layer2D(img: 'Tensorable', f_shape: Union[Iterable, Tuple[int]] = (2, 2), pad: int = 0,
                        stride: int = 2) -> 'Tensor':
    """

    :param img:
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    img = _ensure_tensor(img)

    if (not isinstance(f_shape, Iterable)) and (not isinstance(f_shape, Tuple)):
        raise ValueError(f'could not cast {type(f_shape)} to tuple.')
    if (len(f_shape)) != 2:
        raise ValueError(f'filter\' shape given unexpected length')

    N, C, H, W = img.shape
    fH, fW = f_shape

    if (H + 2 * pad - fH) % stride != 0 or (W + 2 * pad - fW) % stride != 0:
        raise ValueError('dimensions error in meanpooling layer')

    return _meanpooling_tensor4D(img, (fH, fW), pad, stride)


def dropout_layer(t: 'Tensorable', keep_prob: float = 1., axis: int = None, iscorrected: bool = False) -> 'Tensor':
    """
    Example:
    -----------


    Also see:
    -------------
    trchime.nn._core_dropout_layer_tensor

    
    :param iscorrected:
    :param axis:
    :param keep_prob:
    :param t:
    :return: 
    """
    t = _ensure_tensor(t)
    ndim = t.ndim

    if not isinstance(keep_prob, float or int):
        raise TypeError(f'couldn\'t cast {type(keep_prob)} into float or integer')
    if not 1 >= keep_prob >= 0:
        raise ValueError(f'probvalue: {keep_prob}, invalid value for range(0, 1)')
    if not isinstance(axis, int):
        raise TypeError(f'couldn\'t cast {type(axis)} to int')
    if axis >= ndim:
        raise ValueError('given axis out of tensor\'s dimension')
    if not isinstance(iscorrected, bool):
        raise TypeError(f'couldn\'t cast `iscorrected`\'s type: {type(iscorrected)} into bool')

    return _dropout_layer_tensor(t = t, keep_prob = float(keep_prob), axis = axis, iscorrected = iscorrected)
