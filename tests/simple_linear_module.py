
from trchime.core.tensor import Tensor
import trchime as tce
import numpy as np


x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([[-1], [+3], [-2]]))

# set no noise for test
y_data = x_data @ coef + 5 + (np.random.randn(100, 1)) * 1


w = Tensor(np.random.randn(3, 1), requires_grad = True)
b = Tensor(np.random.randn(), requires_grad = True)
learning_rate = 0.01
batch_size = 32

for epoch in range(100):
    epoch_loss = 0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = x_data[start: end]
        predicted = inputs @ w + b
        actual = y_data[start: end]
        # errors =predicted - actual
        # loss = (errors * errors).sum()
        loss = tce.sum((predicted - actual)**2) / 32

        loss.backward()
        epoch_loss += loss.data

        w.assign_sub(learning_rate * w.grad)
        b.assign_sub(learning_rate * b.grad)
        # print(loss.data)
    print(epoch, ":", epoch_loss)

print("w:", w)
print('b:', b)

