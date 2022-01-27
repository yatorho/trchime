import warnings
from typing import Iterator
import inspect

import dill
import numpy as np

from .tensor import Tensor
from .parameter import Parameter
from ..call import ProgressBar
from typing import List, Tuple
# noinspection PyProtectedMember
from .gather import _ensure_tensor


class _Fitter:
    """

    """

    def __init__(self,
                 x: 'Tensor',
                 y: 'Tensor',
                 epoch: int = 1,
                 batch_size: int = None,
                 shuffle: bool = True,
                 validation_split: float = None,
                 validation_data: 'Tuple[Tensor]' = None,
                 validation_freq: int = 1,
                 show_acc_tr: bool = False,
                 show_acc: bool = False):
        """

        :param shuffle:
        :param x:
        :param y:
        :param batch_size:
        :param validation_split:
        :param validation_data:
        :param validation_freq:
        """

        self._shuffle(shuffle, x, y)
        self.batch_size = batch_size
        self.epoch = epoch

        self.notestflag = False

        if validation_data is None:
            if validation_split is not None:
                self.x_test = self.x_data[0: int(self.x_data.shape[0] * validation_split)]
                self.y_test = self.y_data[0: int(self.y_data.shape[0] * validation_split)]

                self.x_data = self.x_data[int(self.x_data.shape[0] * validation_split):]
                self.y_data = self.y_data[int(self.y_data.shape[0] * validation_split):]
            else:
                self.notestflag = True
        else:
            self.x_test = validation_data[0]
            self.y_test = validation_data[1]

        self.validation_freq = validation_freq
        self.show_acc_tr = show_acc_tr
        self.show_acc = show_acc

        self.pd = ProgressBar()

    def _shuffle(self, isshuffle, x_data: 'Tensor', y_data: 'Tensor'):
        """

        :param isshuffle:
        :param x_data:
        :param y_data:
        :return:
        """
        if isshuffle:
            seed = np.random.randint(200)
            np.random.seed(seed)
            np.random.shuffle(x_data.data)
            np.random.seed(seed)
            np.random.shuffle(y_data.data)

            np.random.seed(np.random.randint(300))

            self.x_data = x_data
            self.y_data = y_data


        else:
            self.x_data = x_data
            self.y_data = y_data

    def _show_progress(self, i: int, msg="", f: bool = False):
        """

        :param i:
        :param msg:
        :return:
        """
        if not f:
            print('\r' + self.pd(i) + msg, end = "")
        else:
            print('\r' + msg + self.pd(i), end = "")

    def _mean_square_loss_fit(self, model: 'Module', show_acc_tr: bool, show_acc: bool, freq) -> 'None':
        warnings.warn("`_mean_square_loss_fit` method has been deprecated. take `_fit` instead", DeprecationWarning)
        f = 0

        for epoch in range(self.epoch):
            epoch_loss = 0
            f += 1 / freq

            for start in range(0, self.x_data.shape[0], self.batch_size):
                self._show_progress(100 * start / self.x_data.shape[0], f'{epoch}th epoch', True)

                end = start + self.batch_size

                inputs = self.x_data[start: end]
                acctual = self.y_data[start: end]

                predicted = model.train(inputs)

                self.loss.define_loss(predicted, acctual)
                self.loss.backward()

                self.optimizer.step(model)

                epoch_loss += self.loss.loss.data

            if f >= 1:
                f = 0
                if not self.notestflag:
                    y_test_hat = model.predict(self.x_test)
                    a = y_test_hat.argmax(axis = 1)
                    b = self.y_test.argmax(axis = 1)
                    c = (1 * (a == b)).mean()

                    if show_acc_tr:
                        y_train_hat = model.predict(self.x_data)
                        a1 = y_train_hat.argmax(axis = 1)
                        b1 = self.y_data.argmax(axis = 1)
                        c1 = (1 * (b1 == a1)).mean()
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100),
                              " |  Acc_tr: %5.2f%%" % (c1.data * 100))
                    elif show_acc:
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100))
                    if (not show_acc) and (not show_acc_tr):
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)
                else:
                    print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)

    def _categorical_cross_entropy_loss(self, model: 'Module', show_acc_tr: bool, show_acc, freq) -> 'None':
        """

        :param model:
        :param show_acc_tr:
        :param freq:
        :return:
        """
        warnings.warn("`_categorical_cross_entropy_loss` method has been deprecated. take `_fit` instead",
                      DeprecationWarning)
        f = 0

        for epoch in range(self.epoch):
            epoch_loss = 0
            f += 1 / freq

            for start in range(0, self.x_data.shape[0], self.batch_size):
                self._show_progress(100 * start / self.x_data.shape[0], f'{epoch}th epoch', True)

                end = start + self.batch_size

                inputs = self.x_data[start: end]
                acctual = self.y_data[start: end]

                predicted = model.train(inputs)

                self.loss.define_loss(predicted, acctual)

                self.loss.backward()
                self.optimizer.step(model)

                epoch_loss += self.loss.loss.data

            if f >= 1:
                f = 0
                if not self.notestflag:
                    y_test_hat = model.predict(self.x_test)
                    a = y_test_hat.argmax(axis = 1)
                    b = self.y_test.argmax(axis = 1)
                    c = (1 * (a == b)).mean()

                    if show_acc_tr:
                        y_train_hat = model.predict(self.x_data)
                        a1 = y_train_hat.argmax(axis = 1)
                        b1 = self.y_data.argmax(axis = 1)
                        c1 = (1 * (b1 == a1)).mean()
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100),
                              " |  Acc_tr: %5.2f%%" % (c1.data * 100))
                    elif show_acc:
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100))
                    if (not show_acc) and (not show_acc_tr):
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)
                else:
                    print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)

    def _fit(self, model: 'Module'):
        f = 0

        for epoch in range(self.epoch):
            epoch_loss = 0
            f += 1 / self.validation_freq

            for start in range(0, self.x_data.shape[0], self.batch_size):
                self._show_progress(100 * start / self.x_data.shape[0], f'{epoch}th epoch', True)

                end = start + self.batch_size

                inputs = self.x_data[start: end]
                acctual = self.y_data[start: end]

                predicted = model.train(inputs)

                self.loss.define_loss(predicted, acctual, model)
                self.loss.backward()

                self.optimizer.step(model)

                epoch_loss += self.loss.loss.data

            if f >= 1:
                f = 0
                if not self.notestflag:
                    y_test_hat = model.predict(self.x_test)
                    a = y_test_hat.argmax(axis = 1)
                    b = self.y_test.argmax(axis = 1)
                    c = (1 * (a == b)).mean()

                    if self.show_acc_tr:
                        y_train_hat = model.predict(self.x_data)
                        a1 = y_train_hat.argmax(axis = 1)
                        b1 = self.y_data.argmax(axis = 1)
                        c1 = (1 * (b1 == a1)).mean()
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100),
                              " |  Acc_tr: %5.2f%%" % (c1.data * 100))
                    elif self.show_acc:
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuray:  %5.2f%%" % (c.data * 100))
                    if (not self.show_acc) and (not self.show_acc_tr):
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)
                else:
                    print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)


    def fit(self, model: 'Module', compile):
        from ..nn import SGD_OPTIMIZER, SGDM_OPTIMIZER, ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, RMSPROP_OPTIMIZER
        from ..nn import MSELOSS, CATEGORYLOSS

        optimizer_f = compile.get('optimizer')

        if optimizer_f == SGD_OPTIMIZER:
            self.optimizer = SGD(compile.get('learning_rate', None))

        elif optimizer_f == SGDM_OPTIMIZER:
            self.optimizer = SGDM(compile.get('learning_rate', None), compile.get('sgdm_beta', None))

        elif optimizer_f == RMSPROP_OPTIMIZER:
            self.optimizer = RMSprop(compile.get('learning_rate', None), compile.get('rmsprop_beta', None))

        elif optimizer_f == ADAGRAD_OPTIMIZER:
            self.optimizer = Adagrad(compile.get('learning_rate', None))

        elif optimizer_f == ADAM_OPTIMIZER:
            self.optimizer = Adam(compile.get('learning_rate', None), compile.get('adam_beta1', None),
                                  compile.get('adam_beta2', None), compile.get('adam_corrected', None))

        elif isinstance(optimizer_f, Optimizer):
            self.optimizer = optimizer_f


        loss_f = compile.get('loss', MSELOSS)

        if loss_f == MSELOSS:
            self.loss = MSE_LOSS()
            self._fit(model)

        elif loss_f == CATEGORYLOSS:
            self.loss = CATEGORY_LOSS()
            self._fit(model)

        elif isinstance(loss_f, LOSS):
            self.loss = loss_f
            self._fit(model)


class Module:

    def __init__(self):
        self.layer_manager = Layer_Manager()
        self.sequence = False

    def parameters(self) -> Iterator[Parameter]:

        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                if value.requires_grad:
                    yield value
            elif isinstance(value, Module):
                yield from value.parameters()

        for layer in self.layer_manager.layers_list:
            if isinstance(layer.weight, Tensor):
                if layer.weight.requires_grad:
                    yield layer.weight
            if isinstance(layer.bias, Tensor):
                if layer.bias.requires_grad:
                    yield layer.bias

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def predict(self, inputs) -> 'Tensor':
        if not self.sequence:
            return self.train(inputs)
        else:
            return self.forward(inputs)

    def train(self, inputs) -> 'Tensor':
        if not self.sequence:
            return self.predict(inputs)
        else:
            return self.forward(inputs)

    def compile(self,
                optimizer,
                loss=None,
                metrics=None,
                **kwargs) -> 'None':
        """
        Compile your model in here.
        Examples:
        ---------

        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._compile_file = {'optimizer': self.optimizer,
                              'loss': self.loss,
                              'metrics': self.metrics}
        self._compile_file.update(kwargs)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            validation_split=None,
            validation_data=None,
            shuffle=True,
            validation_freq=1,
            show_acc_tr: bool = False,
            show_acc: bool = False):
        """


        """
        x = _ensure_tensor(x)
        y = _ensure_tensor(y)  # tf

        fitter = _Fitter(x,
                         y,
                         epochs,
                         batch_size,
                         shuffle,
                         validation_split,
                         validation_data,
                         validation_freq,
                         show_acc_tr,
                         show_acc)

        if len(self.layer_manager.layers_list) > 0:
            self.__construct_grap(x.shape)
            self.sequence = True

        fitter.fit(self, self._compile_file)

    def summary(self):
        """
        summary the imformation for your modle.
        :return:
        """
        pass

    def __construct_grap(self, inputs_shape: tuple):
        self.layer_manager.construct_grap(inputs_shape)

    def add(self, layer: 'ConnectionLayer') -> None:
        self.layer_manager.add(layer)

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        return self.layer_manager.forward(inputs, allow_activate)


def savemodel(model: 'Module', url: str = "", *args, **kwargs) -> 'None':
    """
    Herer implements saving operation for model.

    Examples:
    ----------

    :param model:
    :param url:
    :return:
    """

    if not isinstance(model, Module):
        raise RuntimeError("target must be `Module`")
    elif not isinstance(url, str) or url == "":
        raise RuntimeError("invalided url")

    return _savemodel(model, url, *args, **kwargs)


def _savemodel(model: 'Module', url: str, *args, **kwargs):
    """

    :param model:
    :param url:
    :param args:
    :param kwargs:
    :return:
    """

    for tensor in model.parameters():
        tensor.non_depends_on()
        if np.isnan(tensor.data).any():
            raise ValueError("failed to save a model whose parameters has nan value")

    with open(url, 'wb') as file:
        dill.dump(model, file)


def loadmodel(url: str) -> 'Module':
    """

    :param url:
    :return:
    """

    if not isinstance(url, str) or url == "":
        raise RuntimeError("invalided url")

    return _loadmodel(url)


def _loadmodel(url: str) -> 'Module':
    """

    :param url:
    :return:
    """

    with open(url, 'rb') as file:
        model = dill.load(file)

    return model


class GradientTape:
    """
    The GradientTape takes a tensor fn and calls its backward function.
    Morely, GradientTape would zero_grad all variable of fn when closed.
    """

    def __init__(self):
        self.isrun = False

    def gradient(self, fn: Tensor, parameters: List[Tensor], dgrad: Tensor = None):
        self.isrun = True

        fn.backward(dgrad)

        grad: List[Tensor] = []
        for param in parameters:
            grad.append(param.grad)

        return grad

    def _zero_all_grad(self, fn: Tensor):
        if self.isrun:
            raise NotImplementedError
        raise NotImplementedError


from ..nn.optim import SGD, SGDM, RMSprop, Adagrad, Adam, Optimizer
from ..nn.layer import ConnectionLayer, Layer_Manager
from ..nn.loss import LOSS, MSE_LOSS, CATEGORY_LOSS
