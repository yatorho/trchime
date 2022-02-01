import trchime as tce
from trchime.nn import Convolution_layer2D, AveragePool_layer2D, MaxPooling_layer2D, Flatten, Dense
from trchime.nn import Activation
from trchime.datasets import mnist_set

# load train data and test data

url = 'E:/mnist_datasets.npz'

mnist = mnist_set(url)  # pass your dataset url

x_train, y_train, x_test, y_test = mnist.load(whether_tensor = True)  # read mnist dataset, return Tensor data.

y_train, y_test = tce.one_hot(y_train, 10), tce.one_hot(y_test, 10)  # one hot code y sets

x_train, x_test = x_train.reshape((60000, 1, 28, 28)), x_test.reshape((10000, 1, 28, 28))
x_train, x_test = x_train / 255, x_test / 255  # normalize x sets

assert x_train.shape == (60000, 1, 28, 28)
assert y_train.shape == (60000, 10)
assert x_test.shape == (10000, 1, 28, 28)
assert y_test.shape == (10000, 10)


model = tce.Module()  # instantiate a model

# add some computing layer for your model

model.add(Convolution_layer2D(kernel_shape = (3, 3), nums = 3, activation = Activation.RELU_ACTIVATION))
model.add(MaxPooling_layer2D(kernel_shape = (2, 2), stride = 2))
model.add(Convolution_layer2D(kernel_shape = (4, 4), nums = 12, activation = Activation.RELU_ACTIVATION))
model.add(AveragePool_layer2D(kernel_shape = (2, 2), stride = 2))
model.add(Flatten())
model.add(Dense(nums = 45, activation = Activation.TANH_ACTIVATION))
model.add(Dense(nums = 10, activation = Activation.SOFTMAX_ACTIVATION))

# compile your model
model.compile(optimizer=tce.nn.ADAM_OPTIMIZER,  # choose adaptive moment estimation optimizer
              loss=tce.nn.MSELOSS,  # set mean square loss function
              learning_rate=0.001)  # set learning rate

# train your model
model.fit(x_train, y_train,  # input train data
          batch_size=32,  # set batch_size and epochs
          epochs=100,
          validation_data=(x_test, y_test),  # input test data
          show_acc=True,  # show accuracy of test per epoch
          show_acc_tr=True)  # show accuracy of train per epoch
