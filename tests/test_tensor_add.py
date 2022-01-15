from trchime.core.tensor import Tensor
from trchime.core.multitensor import add
import numpy as np


class TestTensorAdd:
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad = True)
        t2 = Tensor([4, 5, 6], requires_grad = True)


        t3 = add(t1, t2)
        t3.backward(Tensor([1, 2, -3]))

        print(t1.grad.data.tolist())
        print(t2.grad.data.tolist())

    def test_broadcast_add(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad = True)  # (3,)

        t3 = add(t1, t2)
        t3.backward(Tensor([[1, 1, 1], [2, 3, 2]]))
        print(t1.grad.data.tolist())
        print(t2.grad.data.tolist())

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad = True)  # (1,3)

        t3 = add(t2, t1)
        t3.backward(Tensor([[1, 1, 1], [2, 2, 2]]))
        print(t1.grad.data)
        print(t2.grad.data)

TestTensorAdd().test_broadcast_add()
