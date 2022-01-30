import numpy as np
from sklearn import datasets
import trchime as tce

wine = datasets.load_wine()
x_input = wine.data[:, :2]  # (178, 2)
y_output = wine.target  # (178,)
print(x_input.shape)
print(y_output.shape)

copy = np.zeros((178, 1))
copy[:, 0] = y_output  # generate (178, 1)

x_input = tce.Tensor(x_input, requires_grad=True)
y_output = tce.Tensor(copy.data, requires_grad=True)  # (178, 1)


# generate train input and output

class Wang_model(tce.Module):
    def __init__(self):
        self.init()
        # initialize paramters' matrix randomly
        self.w1 = tce.random.randn(2, 64, requires_grad=True) * 0.01
        self.b1 = tce.random.randn(1, 64, requires_grad=True) * 0.01

        self.w2 = tce.random.randn(64, 32, requires_grad=True) * 0.01
        self.b2 = tce.random.randn(1, 32, requires_grad=True) * 0.01

        self.w3 = tce.random.randn(32, 16, requires_grad=True) * 0.01
        self.b3 = tce.random.randn(1, 16, requires_grad=True) * 0.01

        self.w4 = tce.random.randn(16, 1, requires_grad=True) * 0.01
        self.b4 = tce.random.randn(1, 1, requires_grad=True) * 0.01

    def predict(self, inputs):
        # define your forward model here
        # 2x64x32x16x1
        z1 = inputs @ self.w1 + self.b1  # (178, 64)
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2  # (178, 32)
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3 + self.b3  # (178, 16)
        a3 = tce.ReLU(z3)

        z4 = a3 @ self.w4 + self.b4  # (178, 1)

        return tce.sigmoid(z4)  # output layer use sigmoid activate function


model = Wang_model()

model.compile(optimizer=tce.nn.SGDM_OPTIMIZER,  # choose stochastic gradient descent optimizer
              loss=tce.nn.MSELOSS,  # set mean square loss function
              learning_rate=10000)  # set learning rate

# train your model
model.fit(x_input, y_output,  # input training data
          batch_size=32,  # set batch_size and epochs
          epochs=200,
          validation_split=0.1,   # split 20% of training set as test set
          show_loss=True,  # show the test loss
          show_acc=True)  # show accuracy per epoch
