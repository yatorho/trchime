
from typing import NamedTuple, Callable
# from . import tensor as t
import numpy as np
from .tensor import Tensor

class Dependency(NamedTuple):

    tensor: 'Tensor'
    grad_fn: 'Callable[[np.ndarray], np.ndarray]'


