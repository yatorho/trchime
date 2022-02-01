# noinspection PyPackageRequirements
import tensorflow as tf
import trchime as tce

fashion = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion.load_data()

x_train = tce.Tensor(x_train).reshape((x_train.shape[0], 1, 28, 28))
y_train = tce.one_hot(tce.Tensor(y_train), 10)
x_test = tce.Tensor(x_test).reshape((x_test.shape[0], 1, 28, 28))
y_test = tce.one_hot(tce.Tensor(y_test), 10)

# normalize data
x_train, x_test = x_train / 255, x_test / 255

class MniModel(tce.Module):

    def __init__(self):
        self.filter1 = tce.random.randn(3, 1, 3, 3, requires_grad = True) * 0.05
        self.bias1 = tce.random.randn(3, 1, requires_grad = True) * 0.05
        self.filter2 = tce.random.randn(12, 3, 4, 4, requires_grad = True) * 0.05
        self.bias2 = tce.random.randn(12, 1, requires_grad = True) * 0.05

        self.w1 = tce.random.randn(300, 45, requires_grad = True) * 0.05
        self.b1 = tce.random.randn(1, 45, requires_grad = True) * 0.05
        self.w2 = tce.random.randn(45, 10, requires_grad = True) * 0.05
        self.b2 = tce.random.randn(1, 10, requires_grad = True) * 0.05


    def predict(self, inputs):
        c1 = tce.nn.Conv2D(inputs, self.filter1, self.bias1)
        p1 = tce.nn.Meanpool2D(c1, (2, 2))
        p1 = tce.ReLU(p1)

        c2 = tce.nn.Conv2D(p1, self.filter2, self.bias2)
        p2 = tce.nn.Meanpool2D(c2, (2, 2))
        p2 = tce.ReLU(p2)

        z1 = p2.reshape((-1, 300)) @ self.w1 + self.b1
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2

        return tce.softmax(z2, axis = 1)

model = MniModel()
model.comrpile(optimizer = 'adam', loss = 'categorical_cross_entropy', learning_rate = 0.01)
model.fit(x_train, y_train, 64, 27, validation_data = (x_test, y_test), show_acc_tr = True)
