import tensorflow as tf
import trchime as tce
from trchime.nn import Dense, AveragePool_layer2D, Convolution_layer2D, Flatten

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tce.Tensor(x_train).reshape((x_train.shape[0], 1, 28, 28))  # 60000, 1, 28, 28
y_train = tce.one_hot(tce.Tensor(y_train), 10)
x_test = tce.Tensor(x_test).reshape((x_test.shape[0], 1, 28, 28))
y_test = tce.one_hot(tce.Tensor(y_test), 10)

x_train, x_test = x_train / 255, x_test / 255

model = tce.Module()

model.add(Convolution_layer2D((3, 3), 3))
model.add(AveragePool_layer2D((2, 2), tce.nn.Activation.RELU_ACTIVATION))
model.add(Convolution_layer2D((4, 4), 12))
model.add(AveragePool_layer2D((2, 2), tce.nn.Activation.RELU_ACTIVATION))
model.add(Flatten())
model.add(Dense(45))
model.add(Dense(10, tce.nn.Activation.SOFTMAX_ACTIVATION))

model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,
              loss = tce.nn.CATEGORYLOSS,
              learning_rate = 0.01)

model.fit(x_train, y_train, 64, 2, validation_data = (x_test, y_test), show_acc_tr = True)

tce.savemodel(model, 'model/mnist_conv_layer.pkl')
