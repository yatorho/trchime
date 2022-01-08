"""

"""
import warnings

from ..core.tensor import Tensor


class layer:

    def __init__(self, name: str, nums: int, activation: str) -> 'None':
        self.name = name
        self.nums = nums
        self.activation = activation
        self.layer_file = {'name': self.name,
                           'nums': self.nums,
                           'activation': self.activation}

    def _layer_file_updata(self, compile_file):
        self.layer_file.update(compile_file)

    def next(self):
        pass


class Linear(layer):

    def __init__(self, nums: int, activation: str = 'relu', **compile_file):
        warnings.warn(DeprecationWarning)
        super().__init__('linear', nums, activation)
        self._layer_file_updata(compile_file)

    def next(self):
        raise NotImplementedError

class Densor(layer):

    def __init__(self, nums: int, activation: str = 'relu', **compile_file):
        super().__init__('densor', nums, activation)
        self._layer_file_updata(compile_file)

    def next(self):
        raise NotImplementedError



class Tanh(layer):
    warnings.warn(DeprecationWarning)
    def __init__(self, nums: int, activation: str = 'relu', **compile_file):
        super().__init__('Tanh', nums, activation)
        self._layer_file_updata(compile_file)

    def next(self):
        raise NotImplementedError


