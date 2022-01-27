from trchime.core.tensor import Tensor

x = Tensor([1, 1, 1], requires_grad = True)
y = x.sum()
y *= 3
print(y)

y.backward()
print(x.grad)
