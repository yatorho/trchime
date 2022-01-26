import trchime as tce
import numpy as np

from trchime.nn.optim import SGD, RMSprop, SGDM, Adagrad


class Mymodule(tce.Module):
    def __init__(self):
        self.x = tce.random.rand(5, require_grad = True)

    def predict(self, inputs):
        return tce.square(self.x - inputs).sum()




cons = tce.Tensor([2, -10, 3, 4, -tce.e])

model = Mymodule()

lr = 0.1
optimizer = RMSprop(lr = lr)

for i in range(2500):
    loss = model.predict(cons)
    loss.backward()
    optimizer.step(module = model)

    print(loss.data)

print(model.x)
