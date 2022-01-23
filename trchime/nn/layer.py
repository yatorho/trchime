from typing import List

from ..core.tensor import Tensor
from ..random.core import randn, normal
from ..core.func import ReLU, ReLUx, tanh, sigmoid, Leaky_ReLU, softplus, softmax, ELU
from .func import Conv2D, Maxpool2D, Meanpool2D
from ..core.multitensor import sqrt


class Activation:
    TANH_ACTIVATION = 'HyperbolicTangent_active_function'
    SIGMOID_ACTIVATION = 'Sigmoid_active_function'
    RELU_ACTIVATION = 'RectifiedLLinearUnit_active_function'
    LEAKY_RELU_ACTIVATION = 'LeakyRectifiedLLinearUnit_active_function'
    SOFTPLUS_ACTIVATION = 'SoftPlus_active_function'
    SOFTMAX_ACTIVATION = 'SoftMax_active_function'
    ELU_ACTIVATION = 'ExponentialLinearUnits_active_function'
    RELUX_ACTIVATION = 'RectifiedLLinearUnitX_active_function'
    NONE = 'NONE'


class Activator:
    def __init__(self, activation: str, **kwargs):
        self.activation = activation

    def activate(self, inputs: Tensor) -> Tensor:
        if self.activation == Activation.TANH_ACTIVATION:
            return tanh(inputs)
        elif self.activation == Activation.SIGMOID_ACTIVATION:
            return sigmoid(inputs)
        elif self.activation == Activation.RELU_ACTIVATION:
            return ReLU(inputs)
        elif self.activation == Activation.LEAKY_RELU_ACTIVATION:
            return Leaky_ReLU(inputs)
        elif self.activation == Activation.SOFTPLUS_ACTIVATION:
            return softplus(inputs)
        elif self.activation == Activation.SOFTMAX_ACTIVATION:
            return softmax(inputs, axis = 1)
        elif self.activation == Activation.ELU_ACTIVATION:
            return ELU(inputs)
        elif self.activation == Activation.RELUX_ACTIVATION:
            return ReLUx(inputs)
        elif self.activation == Activation.NONE:
            return inputs


class Layer_Manager:
    def __init__(self):
        self.layers_list: List[ConnectionLayer] = []

    def add(self, layer: 'ConnectionLayer') -> None:
        self.layers_list.append(layer)

    def construct_grap(self, inputs_shape: tuple):
        next_inputs = inputs_shape

        for i, layer in enumerate(self.layers_list):
            if isinstance(layer, Flatten):
                next_inputs = layer.forward_shape(next_inputs)

            elif len(next_inputs) == 2 and isinstance(layer, Dense):
                layer.input_nums = next_inputs[1]
                layer.output_nums = layer.nums
                next_inputs = (next_inputs[0], layer.nums)

                layer.weight = randn(layer.input_nums, layer.output_nums, requires_grad = True) * 0.1
                layer.bias = randn(1, layer.output_nums, requires_grad = True) * 0.1

            elif len(next_inputs) == 4 and isinstance(layer, Convolution_layer2D):
                chanels_nums = next_inputs[1]

                layer.weight = randn(layer.nums, chanels_nums, layer.kernel_shape[0], layer.kernel_shape[1],
                                     requires_grad = True) * 0.1
                layer.bias = randn(layer.nums, 1, requires_grad = True) * 0.1

                next_inputs = layer.forward_shape(next_inputs)

            elif len(next_inputs) == 4 and isinstance(layer, MaxPooling_layer2D):
                next_inputs = layer.forward_shape(next_inputs)

            elif len(next_inputs) == 4 and isinstance(layer, AveragePool_layer2D):
                next_inputs = layer.forward_shape(next_inputs)

            elif isinstance(layer, Batch_normalize_layer):
                layer.weight = normal(size = (1,) + next_inputs[1:], requires_grad = True)
                layer.bias = normal(size = (1,) + next_inputs[1:], requires_grad = True)




    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        temp_t: Tensor or None = None
        for i, layer in enumerate(self.layers_list):
            if i == 0:
                temp_t = layer.forward(inputs, allow_activate)
            else:
                temp_t = layer.forward(temp_t, allow_activate)

        return temp_t


class ConnectionLayer:
    def __init__(self, name: str = None):
        self.name = name
        self.output_nums: int = 0
        self.input_nums: int = 0
        self.weight: Tensor or None = None
        self.bias: Tensor or None = None

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        pass


class Dense(ConnectionLayer):

    def __init__(self,
                 nums: int = 0,
                 activation: str = Activation.RELU_ACTIVATION):
        super().__init__('denselayer')
        self.nums = nums
        self.activation = activation
        self.activator = Activator(activation)

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            return inputs @ self.weight + self.bias
        else:
            return self.activator.activate(inputs @ self.weight + self.bias)


class Batch_normalize_layer(ConnectionLayer):

    def __init__(self, acivation: str = Activation.NONE):
        super().__init__('batch normalize layer')
        self.activation = acivation
        self.activator = Activator(acivation)

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            mu = inputs.mean(axis = 0, keepdims = True)
            var = inputs.var(axis = 0, keepdims = True)
            t_norm = (inputs - mu) / sqrt(var + 1e-12)
            return self.weight * t_norm + self.bias
        else:
            mu = inputs.mean(axis = 0, keepdims = True)
            var = inputs.var(axis = 0, keepdims = True)
            t_norm = (inputs - mu) / sqrt(var + 1e-12)
            return self.activator.activate(self.weight * t_norm + self.bias)


class Flatten(ConnectionLayer):
    def __init__(self, activation: str = Activation.NONE):
        super().__init__('flattenlayer')
        self.activation = activation
        self.activator = Activator(activation)

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            return inputs.reshape((inputs.shape[0], -1))
        else:
            return self.activator.activate(inputs.reshape((inputs.shape[0], -1)))

    def forward_shape(self, input_shape: tuple) -> tuple:
        if len(input_shape) == 2:
            return input_shape
        else:
            ndims = 1
            for i, n in enumerate(input_shape):
                if i != 0:
                    ndims *= n
            return input_shape[0], ndims


class Convolution_layer2D(ConnectionLayer):
    def __init__(self, kernel_shape: tuple,
                 nums: int,
                 activation: str = Activation.NONE,
                 pad: int = 0,
                 stride: int = 1):
        super().__init__('convolutionallayer2D')
        self.kernel_shape = kernel_shape
        self.nums = nums
        self.activation = activation
        self.activator = Activator(activation)
        self.pad = pad
        self.stride = stride

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            return Conv2D(inputs, self.weight, self.bias, self.pad, self.stride)
        else:
            return self.activator.activate(Conv2D(inputs, self.weight, self.bias, self.pad, self.stride))

    def forward_shape(self, inputs_shape):
        H, W = inputs_shape[2], inputs_shape[3]
        out_h = int(1 + (H + 2 * self.pad - self.kernel_shape[0]) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.kernel_shape[1]) / self.stride)

        return inputs_shape[0], self.nums, out_h, out_w


class MaxPooling_layer2D(ConnectionLayer):
    def __init__(self, kernel_shape: tuple,
                 activation: str = Activation.NONE,
                 pad: int = 0,
                 stride: int = 2):
        super().__init__('maxpoolinglayer2D')
        self.kernel_shape = kernel_shape
        self.activation = activation
        self.activator = Activator(activation)
        self.pad = pad
        self.stride = stride

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            return Maxpool2D(inputs, self.kernel_shape, self.pad, self.stride)
        else:
            return self.activator.activate(Maxpool2D(inputs, self.kernel_shape, self.pad, self.stride))

    def forward_shape(self, inputs_shape):
        H, W = inputs_shape[2], inputs_shape[3]
        out_h = int(1 + (H + 2 * self.pad - self.kernel_shape[0]) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.kernel_shape[1]) / self.stride)

        return inputs_shape[0], inputs_shape[1], out_h, out_w


class AveragePool_layer2D(ConnectionLayer):
    def __init__(self, kernel_shape: tuple,
                 activation: str = Activation.NONE,
                 pad: int = 0,
                 stride: int = 2):
        super().__init__('averagepoolinglayer')
        self.kernel_shape = kernel_shape
        self.activationo = activation
        self.activator = Activator(activation)
        self.pad = pad
        self.stride = stride

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        if not allow_activate:
            return Meanpool2D(inputs, self.kernel_shape, self.pad, self.stride)
        else:
            return self.activator.activate(Meanpool2D(inputs, self.kernel_shape, self.pad, self.stride))

    def forward_shape(self, inputs_shape: tuple) -> tuple:
        H, W = inputs_shape[2], inputs_shape[3]
        out_h = int(1 + (H + 2 * self.pad - self.kernel_shape[0]) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.kernel_shape[1]) / self.stride)

        return inputs_shape[0], inputs_shape[1], out_h, out_w
