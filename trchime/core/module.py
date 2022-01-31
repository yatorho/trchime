import warnings
from typing import Iterator
import inspect

import dill
import numpy as np

from .tensor import Tensor
from .parameter import Parameter, Constant
from ..call import ProgressBar, MessageBoard
from ..call import AccuracyBoard, MultiClassificationAccuracyBoard, MCA_BOARD
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
                 show_acc: bool = False,
                 show_loss: bool = False,
                 epochs_mean_loss: bool = False,
                 show_batch_loss: bool = False,
                 show_batch_acc: bool = False,
                 accuracy_board=MCA_BOARD):
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
        self.show_loss = show_loss
        self.epochs_mean_loss = epochs_mean_loss
        self.show_batch_loss = show_batch_loss
        self.show_batch_acc = show_batch_acc
        self.acc_board_f = accuracy_board

        self.pd = ProgressBar(65)

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
                              " |  Accuracy:  %5.2f%%" % (c.data * 100),
                              " |  Acc_tr: %5.2f%%" % (c1.data * 100))
                    elif show_acc:
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuracy:  %5.2f%%" % (c.data * 100))
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
                              " |  Accuracy:  %5.2f%%" % (c.data * 100),
                              " |  Acc_tr: %5.2f%%" % (c1.data * 100))
                    elif show_acc:
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                              " |  Accuracy:  %5.2f%%" % (c.data * 100))
                    if (not show_acc) and (not show_acc_tr):
                        print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)
                else:
                    print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)

    def _fit(self, model: 'Module'):
        f = 0
        index = 0

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_acc = 0
            f += 1 / self.validation_freq

            for start in range(0, self.x_data.shape[0], self.batch_size):
                self._show_progress(100 * start / self.x_data.shape[0], f'{epoch}th epoch', True)

                end = start + self.batch_size

                inputs = self.x_data[start: end]
                actual = self.y_data[start: end]

                predicted = model.train(inputs)

                self.loss.define_loss(predicted, actual, model)
                self.loss.backward()

                if self.show_batch_loss:
                    print(' |  Loss: %10.5f' % self.loss.loss.data, end = "")

                if self.show_acc_tr:
                    self.accuracy_board.define_accuracy(predicted, actual, model)
                    epoch_acc += self.accuracy_board.accuracy.data

                    if self.show_batch_acc:
                        print(' |  Acc: %5.2f%%' % (self.accuracy_board.accuracy.data * 100), end = "")

                    self.accuracy_board.non_accuracy()

                if self.show_batch_acc or self.show_batch_loss:
                    print()

                self.optimizer.step(model)

                epoch_loss += self.loss.loss.data
                index += 1

            if f >= 1:

                if self.epochs_mean_loss:
                    epoch_loss /= index
                epoch_acc /= index

                f = 0
                index = 0

                if not self.notestflag:
                    y_test_hat = model.predict(self.x_test)
                    self.accuracy_board.define_accuracy(y_test_hat, self.y_test, model)
                    c = self.accuracy_board.accuracy
                    self.accuracy_board.non_accuracy()

                    if not self.show_loss:
                        if self.show_acc_tr:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                                  " |  Accuracy:  %5.2f%%" % (c.data * 100),
                                  " |  Acc_tr: %5.2f%%" % (epoch_acc * 100))

                        elif self.show_acc:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                                  " |  Accuracy:  %5.2f%%" % (c.data * 100))

                        else:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss)

                    else:
                        self.loss.define_loss(y_test_hat, self.y_test, model)

                        if self.show_acc_tr:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                                  " |  loss_test: %12.5f" % self.loss.loss.data,
                                  " |  Accuracy: %5.2f%%" % (c.data * 100),
                                  " |  Acc_tr: %5.2f%%" % (epoch_acc * 100))

                        elif self.show_acc:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                                  " |  loss_test: %12.5f" % self.loss.loss.data,
                                  " |  Accuracy:  %5.2f%%" % (c.data * 100))

                        else:
                            print('\rEpoch: %5d' % epoch, " |  Loss: %12.5f" % epoch_loss,
                                  " |  loss_test: %12.5f" % self.loss.loss.data)

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

        if self.acc_board_f == MCA_BOARD:
            self.accuracy_board = MultiClassificationAccuracyBoard()

        elif isinstance(self.acc_board_f, AccuracyBoard):
            self.accuracy_board = self.acc_board_f

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
        self.init()

    def init(self):
        """
        Here implements the constructor for module.

        You can initialize model's parameters here.

        It's necessary to call `init` method when extend Module

        Example:
        class MyModule(tce.Module):

            def __init__(self):

                self.init()  # It's necessary to call `init` method when extend Module

                self.w1 = tce.Parameter(4, 50)
                self.b1 = tce.Parameter(1, 50)

                self.w2 = tce.Parameter(50, 50)
                self.b2 = tce.Parameter(1, 50)

                self.w3 = tce.Parameter(50, 2)
                self.b3 = tce.Parameter(1, 2)
        """
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

    def constants(self) -> Iterator[Constant]:

        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                if not value.requires_grad:
                    yield value
            elif isinstance(value, Module):
                yield from value.parameters()

        for layer in self.layer_manager.layers_list:
            if isinstance(layer.weight, Tensor):
                if not layer.weight.requires_grad:
                    yield layer.weight
            if isinstance(layer.bias, Tensor):
                if not layer.bias.requires_grad:
                    yield layer.bias

    def __getitem__(self, item):
        val = getattr(self, item, None)
        if isinstance(val, Tensor):
            return val

        elif val is None:
            raise AttributeError(str(self.__class__), "model has no attribute:", item)

        else:
            warnings.warn("called a non-tensor member", FutureWarning)
            return val

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def predict(self, inputs) -> 'Tensor':
        """
        Define the forward propagation of your model here and return model's output
        It's necessary to override `predict` function when calculate the output

        Parameter:
            inputs: 'Tensor', input data
            Return: 'Tensor', output data

        Example:
        class MyModule(tce.Module):

            def __init__(self):

                self.init()

                # randomly initialize parameters
                self.w1 = tce.Parameter(4, 50)
                self.b1 = tce.Parameter(1, 50)

                self.w2 = tce.Parameter(50, 50)
                self.b2 = tce.Parameter(1, 50)

                self.w3 = tce.Parameter(50, 2)
                self.b3 = tce.Parameter(1, 2)

            def predict(self, inputs):  # override `predict` method
                # implements forward propagation for your moel
                z1 = inputs @ self.w1 + self.b1  # (400, 5)
                a1 = tce.ReLU(z1)

                z2 = a1 @ self.w2 + self.b2  # (400, 5)
                a2 = tce.ReLU(z2)

                z3 = a2 @ self.w3 + self.b3  # (400, 2)
                y = tce.sigmoid(z3)
                return y  # return result

        Also see:
            train
            forward

        """
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
                **kwargs) -> 'None':
        """
        Here compiles your model.

        Notes: You can use the compile function to configure your model's optimizer parameters,
               loss function, learning rate, etc.

        Parameters:
        optimizer: a string or a custom optimizer. You can choose an optimizer  or
                   define a new optimizer for you model.

                   Trchime provides five optimizers for your training. You can pass strings of five enum types
                   in tce.nn package as arguments.
                   1. SGD_OPTIMIZER: Stochastic Gradient Descent Optimizer
                   2. SGDM_OPTIMIZER: MomentumOptimizer:
                   3. ADAGRAD_OPTIMIZER: Adaptive Gradient Optimizer
                   4. RMSPROP_OPTIMIZER: Root Mean Square Prop Optimizer
                   5. ADAM_OPTIMIZER: Adaptive Moment Estimation Optimizer

                   You can also customize your own optimizer, which requires you to
                   implement a class that inherits from `Optimizer` in the tce.nn package.
                   You are required to override `__init__` and `step` method of parent class.

        loss: a string or a custom loss. You can choose an loss function or define a new
              loss function for your model.

              Trchime provides three loss function for your training. You can pass strings of three enums types
              in `tce.nn` package as arguments.
              1. MSELOSS: mean_square_error_loss
              1. CATEGORYLOSS: categorical_cross_entropy_loss
              3. MAELOSS: mean_absolute_error_loss

        **kwargs: Optional, You can configure some optional parameters of the optimizer,
                  such as learning_rate, sgdm_beta, adam_corrected, etc.
                  If not given, the optimizer will be initialized with default parameters.
                  You can assign values to the following optional parameters:
                  1. learning_rate:    float, default value is 0.01
                  2. sgm_beta:    float, default value is 0.9
                  3. rmsprop_beta:    float, default value is 0.9
                  4. adam_beta1:    float, default value is 0.9
                  5. adam_beta2:    float, default value is 0.9
                  6. adam_corrected:     boolean, default value is False


        Example:
        ==================================================================================
        model = MyModel()  # instantiate your model
        model.compile(optimizer = tce.nn.SGD_OPTIMIZER,  # choose stochastic gradient descent optimizer
                      loss = tce.nn.MSELOSS,  # set mean square loss function
                      learning_rate = 0.1)  # set learning rate
        ---------------------------------------------------------------------------------


        It's feasible to define a new optimizer

        Example:
        ================================================================================
        class MyOptimizer(tce.nn.Optimizer):

            def __init__(self):
                super().__init__('my optimizer')  # define the name of the new-defined optimizer
                self.lr = 0.1  # set learning rate

            def step(self, module):
                for parameter in module.parameters():
                    parameter.assign_sub(self.lr * parameter.grad)
                    # define the learning way, here is just the gradient descent

        model = MyModel()  # instantiate your model
        my_op = MyOptimizer  # instantiate your optimizer

        model.compile(optimizer = my_op,  # replace with the new optimizer
                      loss = tce.nn.MSELOSS)
        -----------------------------------------------------------------------------------

        Example:
        ================================================================================
        model = MyModel()
        model.compile(optimizer=tce.nn.ADAM_OPTIMIZER,  # choose Adaptive Moment Estimation Optimizer
                      loss=tce.nn.CATEGORYLOSS,  # choose categorical cross entropy loss function
                      learning_rate=0.2,  # set learnin_rate
                      adam_beta1 = 0.95,
                      adam_corrected = True)  # change default arguments for adam optimizer
        --------------------------------------------------------------------------------

        It's feasible to define a new loss function

        Example:
        ===============================================================================
        class MyLoss(tce.nn.loss.LOSS):

            def __init__(self):
                super().__init__("my loss")  # define the name of loss function

            def define_loss(self, predicted: 'Tensor', actual: 'Tensor') -> None:
                '''
                Overrider this method to define your loss function.
                You are required to declare `self.loss`.

                Parameters:
                predicted: The predicted output
                actual: The actual output
                model: training model

                You can get information to define you loss function from above three parameters.

                For examples, your can implements square sum error loss as:
                    >>> self.loss = ((predicted - actual) ** 2).sum()

                '''

                self.loss  = ((predicted-actual)**2).sum()  # define square sum error loss

                for parameter in model.parameters():
                    # model.parameters would return an iterator which contain all trainable parameters in your model
                    self.loss += (parameter ** 2).sum()
                    # implements L1 regularization for your model

        model = MyModel()  # instantiate your model
        my_loss = MyLoss()  # instantiate your loss class

        model.compile(optimizer=tce.nn.ADAM_OPTIMIZER,
                      loss=my_loss,  # replace with new loss
                      learning_rate=0.1)
        ---------------------------------------------------------------------------------
        Also see:
        trchime/examples/xxxx.py

        """

        self._compile_file = {'optimizer': optimizer,
                              'loss': loss}
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
            show_acc: bool = False,
            show_loss: bool = False,
            epochs_mean_loss: bool = False,
            valid_inputs: bool = False,
            show_batch_loss: bool = False,
            show_batch_acc: bool = False,
            accuracy_show=MCA_BOARD):
        """
        Here implements the fit function for model. Your model would be trained from here.

        Parameters:
        x: 'Tensorable', The input of the training set

        y: Tensorable', The output of the training set

        batch_size：optional, 'Integer', batch size.

        epochs: optional, 'Integer', The times of iteration. If not given, default value is 1

        validation_split: optional, 'Float', The proportion of the training set chosen as test set.

        validation_data: optional, 'Tensorable', The test set for neural network which has been trained

        shuffle: optional, 'Bool', Whether to shuffle the inout of training set and test set.
                 If not given, default value is True

        validation_freq: optional, 'Integer', The frequency of showing the accuracy of per epoch.
                         If not given, default value is 1.

        show_acc_tr: optional, 'Bool', Whether to show the accuracy of train set of per epoch. Default value is False.

        show_acc: optional, 'Bool', Whether to show the accuracy of test set of per epoch. Default value is False.

        show_loss: optional, 'Bool', Whether to show the loss of test set of per epoch. Default value is False.

        epochs_mean_loss: optional, 'Bool', Whether to show the average loss of epochs. Default value is False.

        valid_inputs: optional, 'Bool', default value is False.
                      Whether invalidate your input training set. Please ensure the `predict` method of your model
                      never uses the arguments `inputs` before you assign valid_inputs with `True`.

        show_batch_loss: optional, 'Bool', default value is False. Whether show loss per batch.

        show_batch_acc: optional, 'Bool', default value is False. Whether show accuracy per batch.

        accuracy_show: optional, 'AccuracyBoard', default value is MCA_BOARD.
                       Trchime will treat your model as a multi-class network by default
                       and calculate the accuracy of your model output.
                       You can also customize the calculation method of the accuracy of your model output.

        Notes:
        1. Generally, you only need to assign value to anyone of `validation_split` or `validation_data`
           If your assign values to both of them, only `validation_data` would work.'

        2. The frequency of showing accuracy of test set up to once a epoch.

        3. Please ensure the `predict` method of your model never uses the arguments `inputs`
           before you assign valid_inputs with `True`.

        4. More details about customizing your own accuracy's calculation please see:
           trchime/examples/xxxx.py

        Example:
        model.fit(x, y,  # input training data
                  batch_size = 32,  # set batch_size and epochs
                  epochs = 100,
                  validation_split = 0.2,  # split 20% of training set as test set
                  show_acc = True)  # show accuracy per epoch
        More examples please see trchime/examples/

        """
        if not valid_inputs:
            x = _ensure_tensor(x)
            y = _ensure_tensor(y)
        else:
            x = Tensor(1).reshape((1, 1))
            y = Tensor(1).reshape((1, 1))
            batch_size = 1
            shuffle = False
            validation_split = None
            validation_data = None
            validation_freq = 1
            show_acc_tr = False
            show_acc = False

        fitter = _Fitter(x,
                         y,
                         epochs,
                         batch_size,
                         shuffle,
                         validation_split,
                         validation_data,
                         validation_freq,
                         show_acc_tr,
                         show_acc,
                         show_loss,
                         epochs_mean_loss,
                         show_batch_loss,
                         show_batch_acc,
                         accuracy_show)

        if len(self.layer_manager.layers_list) > 0:
            self.__construct_grap(x.shape)
            self.sequence = True

        fitter.fit(self, self._compile_file)

    def summary(self):
        """
        summary the information for your model.
        :return:
        """

        md = MessageBoard(66)

        p_nums = 0
        for p in self.parameters():
            p_nums += p.size

        c_nums = 0
        for c in self.constants():
            c_nums += c.size

        md.add_horizontal_line()
        md.add_text(1, "Model: " + str(self.__class__))

        md.add_horizontal_line(full = 2)

        md.add_horizontal_line()
        md.add_text(3, "Total params: " + str(p_nums + c_nums), width = 66)

        md.add_horizontal_line()
        md.add_text(4, "Trainable params: " + str(p_nums), width = 66)

        md.add_horizontal_line()
        md.add_text(5, "Non-trainable params: " + str(c_nums), width = 66)

        md.add_horizontal_line(full = 1)
        md.show()

    def __construct_grap(self, inputs_shape: tuple):
        self.layer_manager.construct_grap(inputs_shape)

    def add(self, layer: 'ConnectionLayer') -> None:
        """
        Here implements the add function for module.
        This is how the sequence constructs the model.
        You can build a network by adding a forward propagation layer to your network through the `add` method.

        Parameter:
        layer: 'ConnectionLayer': You can define number of neuron in this layer and the activation function

                Trchime provides some computing layers:
                1. Dense:
                2. Batch_normalize_layer:
                3. Flatten:
                4. Convolution_layer:
                5. MaxPooling_layer:
                6. AveragePool_layer:

                Trchime provides some activation function for computing layer in enum class `Activation`:
                1. Activation.TANH_ACTIVATION: hyperbolic Tangent activation function
                2. Activation.SIGMOID_ACTIVATION: sigmoid activation function
                3. Activation.RELU_ACTIVATION: rectified linear unit activation function
                4. Activation.LEAKY_RELU_ACTIVATION: leaky rectified linear unit activation function
                5. Activation.SOFTPLUS_ACTIVATION: softplus activation function
                6. Activation.SOFTMAX_ACTIVATION: softmax activation function
                7. Activation.ELU_ACTIVATION: exponential linear units activation function
                8. Activation.RELUX_ACTIVATION: rectified linear unit x activation function
                9. Activation.NONE: set no activation function

        Example:  # simple network
            model = tce.Module()
            model.init()
            model.add(tce.nn.Dense(nums = 32,  # number of the neuron
                                   activation = tce.nn.Activation.RELU_ACTIVATION))  # define relu activation function

            model.add(tce.nn.Dense(nums = 4,  # number of the neuron
                                   activation = tce.nn.Activation.SOFTMAX_ACTIVATION))  # define softmax activation function

        More examples: trchime/examples/4x32x24x2 network.py
                       trchime/examples/xxxx.py

        """
        self.layer_manager.add(layer)

    def forward(self, inputs: Tensor, allow_activate: bool = True) -> Tensor:
        return self.layer_manager.forward(inputs, allow_activate)


def savemodel(model: 'Module', url: str = "", *args, **kwargs) -> 'None':
    """
    Here implements saving operation for model.

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
    model._compile_file = None

    for tensor in model.parameters():
        tensor.non_depends_on()
        if np.isnan(tensor.data).any():
            raise ValueError("failed to save a model whose parameters has nan value")

    for tensor in model.constants():
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
