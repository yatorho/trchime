from trchime.core.tensor import Tensor
from trchime.core.multitensor import sum

class TestTensorSum:
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad = True)
        t2 = t1.sum()

        t2.backward()

        print(t1.grad.data.tolist())

    def test_sum_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad = True)
        # t2 = sum(t1)
        t2 = t1.sum()

        t2.backward(Tensor(1))

        print(t1.grad.data.tolist())

TestTensorSum().test_sum_with_grad()