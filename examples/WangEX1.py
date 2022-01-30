import numpy as np
from sklearn import datasets
import trchime as tce

iris = datasets.load_iris()
x_input = iris.data[:, :2]  # (150, 2)
y_output = iris.target  # (150,)

copy = np.zeros((150, 1))
copy[:, 0] = y_output  # generate (150, 1)

x_input = tce.Tensor(x_input, requires_grad=True)
y_output = tce.Tensor(copy.data, requires_grad=True)  # (150, 1)

# generate train input and output


class Wang_model(tce.Module):
    def __init__(self):
        self.init()
        # initialize paramters' matrix randomly
        self.w1 = tce.random.randn(2, 32, requires_grad=True) * 0.1
        self.b1 = tce.random.randn(1, 32, requires_grad=True) * 0.1

        self.w2 = tce.random.randn(32, 16, requires_grad=True) * 0.1
        self.b2 = tce.random.randn(1, 16, requires_grad=True) * 0.1

        self.w3 = tce.random.randn(16, 1, requires_grad=True) * 0.1
        self.b3 = tce.random.randn(1, 1, requires_grad=True) * 0.1

    def predict(self, inputs):
        # define your forward model here
        # 2x32x16x1
        z1 = inputs @ self.w1 + self.b1  # (150, 32)
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2  # (150, 16)
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3 + self.b3  # (150, 1)
        return tce.sigmoid(z3)  # output layer use sigmoid activate function


model = Wang_model()

model.compile(optimizer=tce.nn.ADAM_OPTIMIZER,  # choose stochastic gradient descent optimizer
              loss=tce.nn.MSELOSS,  # set mean square loss function
              learning_rate=0.01)  # set learning rate

# train your model
model.fit(x_input, y_output,  # input training data
          batch_size=1,  # set batch_size and epochs
          epochs=200,
          validation_split=0.1,  # split 10% of training set as test set
          show_loss=True,  # show the test loss
          show_acc=True)  # show accuracy per epoch




