import trchime as tce
from trchime.datasets import cifar_set
from trchime.nn import Conv2D, Maxpool2D
from trchime import Module


url = '.E:/cifar_datasets.npz'
save_path = 'cifar_model.pkl'

cifar = cifar_set(url)  # pass your dataset url

x_train, y_train, x_test, y_test = cifar.load(whether_tensor = True)  # read cifar dataset, return Tensor data.

y_train, y_test = y_train.reshape((50000,)), y_test.reshape((10000,))  # reshape y sets
y_train, y_test = tce.one_hot(y_train, 100), tce.one_hot(y_test, 100)  # one hot code y sets

x_train, x_test = x_train / 255, x_test / 255  # normalize x sets


assert x_train.shape == (50000, 3, 32, 32)
assert x_test.shape == (10000, 3, 32, 32)
assert y_train.shape == (50000, 100)
assert y_test.shape == (10000, 100)

class Cifar_model(Module):

    def __init__(self):
        self.init()
        # initialize parameter matrix randomly
        # shape of kernels in `Trchime` would be (N, C, H, W)
        # shape of biases in `Trchime` would be (N, 1)
        self.filter1 = tce.random.randn(12, 3, 3, 3, requires_grad=True) * 0.1
        self.bias1 = tce.random.randn(12, 1, requires_grad=True) * 0.1
        self.filter2 = tce.random.randn(24, 12, 4, 4, requires_grad=True) * 0.1
        self.bias2 = tce.random.randn(24, 1, requires_grad=True) * 0.1

        self.w1 = tce.random.randn(864, 1080, requires_grad=True) * 0.1
        self.b1 = tce.random.randn(1, 1080, requires_grad=True) * 0.1
        self.w2 = tce.random.randn(1080, 100, requires_grad=True) * 0.1
        self.b2 = tce.random.randn(1, 100, requires_grad=True) * 0.1

    def predict(self, inputs):
        # describe forward propagation here

        c1 = Conv2D(inputs, self.filter1, self.bias1)  # (None, 3, 30, 30)
        c1 = tce.ReLU(c1)
        p1 = Maxpool2D(c1, (2, 2))  # None, 12, 15, 15

        c2 = Conv2D(p1, self.filter2, self.bias2)  # (None, 24, 12, 12)
        c2 = tce.ReLU(c2)
        p2 = Maxpool2D(c2, (2, 2))  # None, 24, 6, 6

        # None, 216
        z1 = p2.reshape((-1, 864)) @ self.w1 + self.b1  # None, 1080
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2  # (None, 100)
        a2 = tce.softmax(z2, axis=1)

        return a2



model = Cifar_model()  # instantiate your model

# compile your model
model.compile(optimizer=tce.nn.ADAM_OPTIMIZER,  # choose adaptive moment estimation optimizer
              loss=tce.nn.CATEGORYLOSS,  # set categorical cross entropy loss
              learning_rate=0.005)  # set learning rate

# train your model
model.fit(x_train, y_train,  # input train data
          batch_size=64,  # set batch_size and epochs
          epochs=10,
          validation_data=(x_test, y_test),  # input test data
          show_acc=True,  # show accuracy of test per epoch
          show_acc_tr=True,  # show accuracy of train per epoch
          show_batch_acc = True,  # show accuracy of train per batch
          show_batch_loss = True)  # show loss of train per batch

tce.savemodel(model, save_path)  # save your model
