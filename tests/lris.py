from sklearn import datasets
import numpy as np

import trchime as tce
from trchime.nn.optim import SGD

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tce.Tensor(x_train)
y_train = tce.Tensor(y_train)
x_test = tce.Tensor(x_test)
y_test = tce.Tensor(y_test)

y_train = tce.one_hot(y_train, 3)
y_test = tce.one_hot(y_test, 3)


class Modle(tce.Module):
    def __init__(self):
        self.w = tce.Parameter(4, 3)

        self.b = tce.Parameter(1, 3)


    def predict(self, input):
        y = input @ self.w + self.b
        return tce.softmax(y, axis = 1)



batch_size = 32

model = Modle()
learning_rate = 0.1
optimizer = SGD(lr = learning_rate)


for epoch in range(500):
    epoch_loss = 0

    for start in range(0, 120, batch_size):
        end = start + batch_size

        inputs = x_train[start: end]
        predicted = model.predict(inputs)

        actual = y_train[start: end]

        loss = tce.sum((actual - predicted)**2) / actual.shape[0]
        epoch_loss += loss.data

        loss.backward()

        optimizer.step(model)


    y_test_hat = model.predict(x_test)
    a = y_test_hat.argmax(axis = 1)
    b = y_test.argmax(axis = 1)

    c = (1*(a == b)).sum()

    y_train_hat = model.predict(x_train)
    a1 = y_train_hat.argmax(axis = 1)
    b1 = y_train.argmax(axis = 1)
    c1 = (1*(b1==a1)).sum()

    print(epoch, ":", epoch_loss / 4, ",", c1.data / x_train.shape[0], c.data / x_test.shape[0])
