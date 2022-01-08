"""
Core implementations for function
"""
import numpy as np

from ..core.tensor import Tensor
from ..core.dependency import Dependency

from typing import List


def _img_to_col_iter(img: 'np.ndarray', filter_h: int, filter_w: int, stride: int, padding: int) -> 'np.ndarray':
    """

    Parameters
    ----------
    img :
    filter_h :
    filter_w :
    stride :
    padding :

    Returns
    -------
    col :
    """
    N, C, H, W = img.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype = img.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col


def _col_to_img_iter(col: 'np.ndarray', input_shape, f_shape, stride=1, pad=0) -> 'np.ndarray':
    """

    Parameters
    ----------
    col :
    input_shape :
    f_shape
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    filter_h, filter_w = f_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape((N, out_h, out_w, C, filter_h, filter_w)).transpose((0, 3, 4, 5, 1, 2))

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype = col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

# @jit
def _get_img_to_col_indices(x_shape, field_height: int, field_width: int, padding: int, stride: int) -> 'tuple':
    """
    Figure out what the size of the output should be.
    :param x_shape:
    :param field_height:
    :param field_width:
    :param padding:
    :param stride:
    :return:
    """
    N, C, H, W = x_shape

    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j

# @jit
def _img_to_col_indices(x, field_height: int, field_width: int, padding: int, stride: int) -> 'np.ndarray':
    """
    An implementation of im2col based on some fancy indexing.
    :param x:
    :param field_height:
    :param field_width:
    :param padding:
    :param stride:
    :return:
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

    k, i, j = _get_img_to_col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 0, 2).reshape(field_height * field_width * C, -1)
    # cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols


def __convolution_forward_array2D(img: 'np.ndarray', filter: 'np.ndarray', bias: 'np.ndarray', pad: int,
                                  stride: int) -> 'np.ndarray':
    """

    :param img:
    :param filter:
    :param bias:
    :param pad:
    :param stride:
    :return:
    """

    N, C, H, W = img.shape
    fN, fC, fH, fW = filter.shape

    out_h = int(1 + (H + 2 * pad - fH) / stride)
    out_w = int(1 + (W + 2 * pad - fW) / stride)

    # col = _img_to_col_indices(img, fH, fW, pad, stride).T
    # col_w = filter.reshape((fN, -1)).T
    # col_b = bias.reshape((fN, 1)).T
    #
    # out = np.dot(col, col_w) + col_b
    # out = out.reshape((N, out_h, out_w, -1)).transpose(0, 3, 1, 2)  # N, -1, out_h, out_w

    col = _img_to_col_indices(img, fH, fW, pad, stride)
    col_w = filter.reshape((fN, -1))
    col_b = bias.reshape((fN, 1))

    out = np.dot(col_w, col) + col_b
    out = out.reshape((-1, N, out_h, out_w)).transpose(1, 0, 2, 3)  # N, -1, out_h, out_w

    return out


def __convolution_backward_array_filter2D(grad: 'np.ndarray', img: 'np.ndarray', pad: int, stride: int) -> 'np.ndarray':
    """
    Also see:
    ----------
    _get_img_to_col_indices
    _img_to_col_indices
    __convolution_forward_array4D

    :param grad:
    :param img:
    :param pad:
    :param stride:
    :return:
    """

    if stride != 1:
        grad = __pad_assertwith_0_array4D(grad, stride - 1)

    gN, gC, gH, gW = grad.shape
    N, C, H, W = img.shape

    out_H = int(1 + H + 2 * pad - gH)
    out_W = int(1 + W + 2 * pad - gW)

    col_img = _img_to_col_indices(img.transpose((1, 0, 2, 3)), gH, gW, pad, 1)
    col_g = grad.transpose((1, 0, 2, 3)).reshape((gC, -1))

    out = np.dot(col_g, col_img)
    out = out.reshape((-1, C, out_H, out_W))  # -1, C, cout_H, out_W

    return out


def __convolution_backward_array_img2D(grad: 'np.ndarray', filter: 'np.ndarray', pad: int, stride: int) -> 'np.ndarray':
    """
    Also see:
    ----------
    img * filter + bias = out

    out_grad
    1. stride == 1: img_grad = out_grad(0 padded) * filter_revserse
    2. stride != 1: img_grad = out_grad(insert 0 , 0 padded) * filter_reverse

    out_grad: insert (stride - 1) 0.
    [[1, 0, 2, 0, 3],
     [0, 0, 0, 0, 0],
     [2, 0, 3, 0, 4]]



    :param grad:
    :param filter:
    :param pad:
    :param stride:
    :return:
    """

    filter_reved = filter[:, :, ::-1, ::-1]  # reverse filter for last 2 dimensions

    fN, fC, fH, fW = filter_reved.shape

    p_h = fH - 1 - pad
    p_w = fW - 1 - pad

    if stride != 1:
        grad = __pad_assertwith_0_array4D(grad, stride - 1)

    grad_padded = np.pad(grad, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    N, C, H, W = grad_padded.shape

    out_N, out_C, out_H, out_W = N, fN, H - fH + 1, W - fW + 1

    col_grad = _img_to_col_indices(grad_padded, fH, fW, 0, 1)
    col_f = filter_reved.transpose((1, 0, 2, 3)).reshape((fC, -1))

    out = np.dot(col_f, col_grad)
    out = out.reshape((-1, N, out_H, out_W)).transpose((1, 0, 2, 3))  # N, -1, out_H, out_W

    return out


def __convolution_backward_array_bias2D(grad: 'np.ndarray') -> 'np.ndarray':
    """
    Also see:
    ----------

    :param grad:
    :return:
    """
    N, C, H, W = grad.shape

    grad = grad.transpose((1, 0, 2, 3)).reshape((C, -1))

    out = np.sum(grad, axis = 1, keepdims = True)

    return out


def __pad_assertwith_0_array4D(grad: 'np.ndarray', pad_nums) -> 'np.ndarray':
    """
    Padding arrary with 0 septally.
    :param grad:
    :param pad_nums:
    :return:
    """
    gN, gC, gH, gW = grad.shape

    init1 = np.zeros((gN, gC, gH + (gH - 1) * pad_nums, gW), dtype = grad.dtype)
    init2 = np.zeros((gN, gC, gH + (gH - 1) * pad_nums, gW + (gW - 1) * pad_nums), dtype = grad.dtype)

    boolean: List[int] = [(pad_nums + 1) * i for i in range(grad.shape[2])]
    init1[:, :, boolean, :] = grad

    boolean: List[int] = [(pad_nums + 1) * i for i in range(grad.shape[3])]
    init2[:, :, :, boolean] = init1

    return init2


def _conv_tensor2D(img: 'Tensor', filter: 'Tensor', bias: 'Tensor', pad: int, stride: int) -> 'Tensor':
    """

    :param img:
    :param filter:
    :param bias:
    :param pad:
    :param stride:
    :return:
    """

    data = __convolution_forward_array2D(img.data, filter.data, bias.data, pad, stride)
    requires_grad = img.requires_grad or filter.requires_grad or bias.requires_grad

    depends_on: List[Dependency] = []

    if img.requires_grad:
        """
        
        """

        def grad_f1(grad: 'np.ndarray') -> 'np.ndarray':
            """
            See:
            ------
            __convolution_backward_array_img4D
            """
            return __convolution_backward_array_img2D(grad, filter.data, pad, stride)

        depends_on.append(Dependency(tensor = img, grad_fn = grad_f1))

    if filter.requires_grad:
        """
        """

        def grad_f2(grad: 'np.ndarray') -> 'np.ndarray':
            """
            See:
            -------
            _convolution_backward_array_filter4D
            """
            return __convolution_backward_array_filter2D(grad, img.data, pad, stride)

        depends_on.append(Dependency(tensor = filter, grad_fn = grad_f2))

    if bias.requires_grad:
        """
        """

        def grad_f3(grad: 'np.ndarray') -> 'np.ndarray':
            """
            See:
            -------
            __convolution_backward_array_bias

            """
            return __convolution_backward_array_bias2D(grad)

        depends_on.append(Dependency(tensor = bias, grad_fn = grad_f3))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def __maxpooling_forward_array2D(img: 'np.ndarray', f_shape, pad: int, stride: int) -> 'np.ndarray' or tuple:
    """


    :param img:
    :param pad:
    :param stride:
    :return:
    """
    N, C, H, W = img.shape
    fH, fW = f_shape
    out_h = int(1 + (H + 2 * pad - fH) / stride)
    out_w = int(1 + (W + 2 * pad - fW) / stride)

    col = _img_to_col_indices(img, fH, fW, pad, stride)  # C*fH*fW, N*out_h*out_w
    col = col.reshape((C, fH * fW, N * out_h * out_w))

    max_arg = np.argmax(col, axis = 1)  # C, N * out_h * out_w
    max_arg = max_arg.reshape((C, N, out_h, out_w))  # C, N, out_h, out_w

    out = np.max(col, axis = 1, keepdims = True).reshape((C, N, out_h, out_w))
    out = out.transpose((1, 0, 2, 3))  # N, C, out_h, out_w

    return out, max_arg


def __maxpooling_backward_array2D(grad: 'np.ndarray', max_arg: 'np.ndarray', f_shape, pad: int,
                                  stride: int) -> 'np.ndarray':
    """

    :param grad:  # N, C, h, w
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    fH, fW = f_shape
    N, C, h, w = grad.shape
    out_h = (h - 1) * stride + fH - 2 * pad
    out_w = (w - 1) * stride + fW - 2 * pad

    max_arg = max_arg.transpose((1, 0, 2, 3))  # N, C, h, w
    out = np.zeros((grad.size, fH * fW), dtype = grad.dtype)  # N * C * h * w, fH * fW

    out[np.arange(max_arg.size), max_arg.flatten()] = grad.flatten()
    # out[[0, 1, ..., N * C * h * w - 1], arg_max.flatten] = grad

    out = out.reshape((N, C, h * w, fH * fW))  # N, C, h * w, fH * fW
    out = out.transpose((0, 2, 1, 3))  # N, h * w, C, fH * fW
    out = out.reshape((N * h * w, C * fH * fW))  # N * h * w, C * fH * fW

    out = _col_to_img_iter(out, (N, C, out_h, out_w), (fH, fW), stride, pad)

    return out


def _maxpooling_tensor2D(img: 'Tensor', f_shape, pad: int, stride: int) -> 'Tensor':
    """

    :param img:
    :param pad:
    :param stride:
    :return:
    """
    data, max_arg = __maxpooling_forward_array2D(img.data, f_shape, pad, stride)

    requires_grad = img.requires_grad

    if requires_grad:

        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            """
            See:
            ------
            __maxpooling_backward_array4D

            :param grad:
            :return:
            """
            return __maxpooling_backward_array2D(grad, max_arg, f_shape, pad, stride)

        depends_on: List[Dependency] = [Dependency(tensor = img, grad_fn = grad_f)]
    else:
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def __meanpooling_forward_array2D(img: 'np.ndarray', f_shape, pad: int, stride: int) -> 'np.ndarray':
    """

    :param img:
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    N, C, H, W = img.shape
    fH, fW = f_shape
    out_h = int(1 + (H + 2 * pad - fH) / stride)
    out_w = int(1 + (W + 2 * pad - fW) / stride)

    col = _img_to_col_indices(img, fH, fW, pad, stride)  # C*fH*fW, N*out_h*out_w
    col = col.reshape((C, fH * fW, N * out_h * out_w))  # C, fH*fW, N*out_h*out_w

    out = np.mean(col, axis = 1)  # C, N*out_h*out_w
    out = out.reshape((C, N, out_h, out_w))  # C, N, out_h, out_w

    return out.transpose((1, 0, 2, 3))  # N, C, out_h, out_w


def __meanpooling_backward_array2D(grad: 'np.ndarray', f_shape, pad: int, stride: int) -> 'np.ndarray':
    """

    :param grad:
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    fH, fW = f_shape
    N, C, h, w = grad.shape  # N, C, h, w
    out_h = (h - 1) * stride + fH - 2 * pad
    out_w = (w - 1) * stride + fW - 2 * pad

    out = np.zeros((grad.size, fH * fW), dtype = grad.dtype)  # N*C*h*w, fH*fW
    out[np.arange(grad.size), :] = grad.flatten().reshape(grad.size, 1) / (fH * fW)  # N*C*h*w

    out = out.reshape((N, C, h * w, fH * fW))  # N, C, h * w, fh*fw
    out = out.transpose((0, 2, 1, 3))  # N, h*w, C, fh*fW
    out = out.reshape((N * h * w, C * fH * fW))  # N*h*w, C*fH*fW

    out = _col_to_img_iter(out, (N, C, out_h, out_w), (fH, fW), stride, pad)

    return out


def _meanpooling_tensor4D(img: 'Tensor', f_shape, pad: int, stride: int) -> 'Tensor':
    """

    :param img:
    :param f_shape:
    :param pad:
    :param stride:
    :return:
    """
    data = __meanpooling_forward_array2D(img.data, f_shape, pad, stride)
    requires_grad = img.requires_grad

    if requires_grad:
        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            """
            See:
            ------
            __meanpooling_backward_array4D

            :param grad:
            :return:
            """
            return __meanpooling_backward_array2D(grad, f_shape, pad, stride)

        depends_on: List[Dependency] = [Dependency(tensor = img, grad_fn = grad_f)]
    else:
        depends_on: List[Dependency] = []


    return Tensor(data,
                  requires_grad,
                  depends_on)



def _dropout_layer_tensor(t: 'Tensor', keep_prob: float, axis: int, iscorrected: bool) -> 'Tensor':
    """
    32 * 28 * 28
    32 * 784
    32 * 128
    128 * 32



    :param t:  # (2, 3, 4, 5)
    :param axis:  2
    :param keep_prob:
    :param iscorrected:
    :return:
    """
    length = t.shape[axis]
    k_list = 1 * (np.random.random(length) < keep_prob)

    s = []
    for i in range(t.ndim):
        if i != axis:
            s.append(1)
        else:
            s.append(length)

    k_list = k_list.reshape(tuple(s))  # (1, 1, 4, 1)

    data = t.data * k_list
    if iscorrected:
        data = data / keep_prob

    requires_grad = t.requires_grad

    if requires_grad:

        def grad_f(grad: 'np.ndarray') -> 'np.ndarray':
            """
            grad:
            :param grad:
            :return:
            """
            if iscorrected:
                return grad * k_list / keep_prob
            else:
                return grad * k_list

        depends_on: List[Dependency] = [Dependency(tensor = t, grad_fn = grad_f)]
    else:
        depends_on: List[Dependency] = []

    return Tensor(data,
                  requires_grad,
                  depends_on)
