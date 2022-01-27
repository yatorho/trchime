from ..core.tensor import Tensor
# noinspection PyProtectedMember
from ..core._multitensor import _sum, _mean
# noinspection PyProtectedMember
from ..core._func import _log, _abs
from ..core.module import Module

MSELOSS = 'mean_square_error_loss'
CATEGORYLOSS = 'categorical_cross_entropy_loss'
MAELOSS = 'mean_absolute_error_loss'



class LOSS:

    def __init__(self, name: str = None):
        self.name = name
        self.loss: Tensor or None = None

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', pam: 'Module' = None) -> None:
        pass

    def backward(self):
        self.loss.backward()


class MSE_LOSS(LOSS):

    def __init__(self):
        super().__init__(MSELOSS)

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', pam: 'Module' = None) -> None:
        self.loss = _sum((predicted - actual) ** 2, axis = predicted.ndim - 1, keepdims = True)
        self.loss = _mean(self.loss)


class CATEGORY_LOSS(LOSS):

    def __init__(self):
        super().__init__(CATEGORYLOSS)

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', pam: 'Module' = None) -> None:
        self.loss = _sum(-(actual * _log(predicted)))


class MAE_LOSS(LOSS):

    def __init__(self):
        super().__init__(MAELOSS)

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', pam: 'Module' = None) -> None:
        self.loss = _sum(_abs(predicted - actual, isnew = False), axis = predicted.ndim - 1, keepdims = True)
        self.loss = _mean(self.loss)

