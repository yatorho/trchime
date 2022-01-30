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


class MniModel(tce.Module):

    def __init__(self):
        self.init()
        self.w1 = tce.random.randn(784, 128, requires_grad = True) * 0.01
        # w1. non-depedns_on
        self.b1 = tce.random.randn(1, 128, requires_grad = True) * 0.01

        self.w2 = tce.random.randn(128, 10, requires_grad = True) * 0.01
        self.b2 = tce.random.randn(1, 10, requires_grad = True) * 0.01

    def train(self, inputs):
        z1 = tce.matmul(inputs, self.w1) + self.b1
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2

        return tce.softmax(z2, axis = 1)
        # return tce.sigmoid(z2)

    def predict(self, inputs):
        return self.train(inputs)

model = MniModel()
model.compile(optimizer = tce.nn.ADAM_OPTIMIZER, loss = tce.nn.CATEGORYLOSS, learning_rate = 0.005)

model.fit(x_train, y_train, 64, 1, validation_data = (x_test, y_test), show_acc = True, show_acc_tr = True)

tce.savemodel(model, 'model/mnist_model.pkl')


