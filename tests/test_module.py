from random import random

import trchime as tce
import numpy as np


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

class MyModule(tce.Module):

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

model = MyModule()

model.compile(optimizer = 'adam', loss = 'square_loss', learning_rate = 0.01)

model.fit(x, y, 32, 10, validation_split = 0.3)



