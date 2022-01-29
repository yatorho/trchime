import numpy as np
from .tensor import Tensor
from .gather import Arrayable, _ensure_array

class Parameter(Tensor):
    def __init__(self, *shape) -> 'None':

        data = (np.random.randn(*shape)) * 0.1
        super().__init__(data, requires_grad = True)

class Variable(Tensor):
    def __init__(self, data: Arrayable = None, shape=None):
        if data is None:
            data = (np.random.randn(*shape)) * 0.1
            super().__init__(data, requires_grad = True)

        else:
            super().__init__(_ensure_array(data), requires_grad = True)


class Constant(Tensor):
    def __init__(self, data: Arrayable = None, shape=None):
        if data is None:
            data = (np.random.randn(*shape)) * 0.1
            super().__init__(data)

        else:
            super().__init__(_ensure_array(data))

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> 'None':
        import warnings
        warnings.warn('try to set value of constant', FutureWarning)
        self._data = new_data
