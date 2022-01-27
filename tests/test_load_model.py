# noinspection PyPackageRequirements
import tensorflow as tf
import trchime as tce

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tce.Tensor(x_train).reshape((x_train.shape[0], -1))

"""
function: one_hot
let outputs data mapping higher dimension, such as 3 -> (0, 0, 0, 1)

examples:
>>> a = tce.Tensor([2, 1, 0, 3])
>>> out = tce.one_hot(a, 3)
>>> a
    Tensor([[0, 0, 1, 0]
            [0, 1, 0, 0]
            [1, 0, 0, 0]
            [0, 0, 0, 1]], requires_grad = False, dtype = tce.int32)

"""
y_train = tce.one_hot(tce.Tensor(y_train), 10)

x_test = tce.Tensor(x_test).reshape((x_test.shape[0], -1))

y_test = tce.one_hot(tce.Tensor(y_test), 10)

# normalize data
x_train, x_test = x_train / 255, x_test / 255

model = tce.loadmodel("model/mnist_model.pkl")

model.compile(optimizer = 'adam', loss = 'cross_entropy_loss', learning_rate = 0.005)

model.fit(x_train, y_train, 64, 5, validation_data = (x_test, y_test))

tce.savemodel(model, 'model/mnist_model.pkl')

