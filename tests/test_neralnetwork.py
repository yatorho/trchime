
import trchime as tce
from random import random
import numpy as np

from trchime import Module
from trchime.nn.optim import SGD, Adagrad, SGDM, Adam
from trchime.nn.optim import RMSprop


def getSource():
    examples = 30000
    InputVector = tce.Tensor(np.zeros((examples, 4)))

    OutVector = tce.Tensor(np.zeros((examples, 2)))
    for i in range(examples):
        x = 2 * (random() - 0.5)
        y = 2 * (random() - 0.5)
        z = 2 * (random() - 0.5)
        v = 2 * (random() - 0.5)

        InputVector[i, 0] = x
        InputVector[i, 1] = y
        InputVector[i, 2] = z
        InputVector[i, 3] = v

        if x ** 2 + y ** 2 + z ** 2 + v ** 2 < 1:
            OutVector[i, 0] = 1
            OutVector[i, 1] = 0
        else:
            OutVector[i, 0] = 0
            OutVector[i, 1] = 1
    return InputVector, OutVector


class Modle(Module):
    def __init__(self):
        self.w1 = tce.Parameter(4, 50)
        self.b1 = tce.Parameter(1, 50)

        self.w2 = tce.Parameter(50, 50)
        self.b2 = tce.Parameter(1, 50)

        self.w3 = tce.Parameter(50, 2)
        self.b3 = tce.Parameter(1, 2)

    def predict(self, inputs):
        z1 = inputs @ self.w1 + self.b1  # (400, 5)
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2  # (400, 5)
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3 + self.b3  # (400, 2)
        y = tce.sigmoid(z3)
        return y
        # return tce.softmax(y, axis = 1)


x, y = getSource()
x_data = x[:20000]
y_data = y[:20000]
x_test = x[20000:]
y_test = y[20000:]

batch_size = 32

model = Modle()
learning_rate = 0.01
# optimizer = Adam(learning_rate, corrected = True)
optimizer = Adam(learning_rate)

for epoch in range(10):
    epoch_loss = 0

    for start in range(0, 20000, batch_size):
        end = start + batch_size

        inputs = x_data[start: end]
        predicted = model.predict(inputs)

        actual = y_data[start: end]
        # loss = - actual * ad.log(predicted) - (1-actual)*ad.log(1-predicted)
        # loss = loss.sum() / actual.shape[0]
        # loss = ((actual - predicted)**2).sum() / actual.shape[0]
        loss = tce.mean((actual - predicted) ** 2, axis = 1, keepdims = True)
        loss = loss.sum()

        loss.backward()
        epoch_loss += loss.data

        optimizer.step(model)

    y_test_hat = model.predict(x_test)
    a = y_test_hat.argmax(axis = 1)
    b = y_test.argmax(axis = 1)

    c = (1 * (a == b)).sum()

    y_train_hat = model.predict(x_data)
    a1 = y_train_hat.argmax(axis = 1)
    b1 = y_data.argmax(axis = 1)
    c1 = (1 * (b1 == a1)).sum()

    print(epoch, ":", epoch_loss, ",", c1.data / x_data.shape[0], c.data / x_test.shape[0])

print(model.predict(tce.Tensor([[0.1, 0.3, -tce.e, 0.2]])))