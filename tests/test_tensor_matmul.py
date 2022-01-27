from trchime.core.tensor import Tensor


class TestTensorMul:
    def test_simple_matmul(self):
        t1 = Tensor([[1, 2, 3]], requires_grad = True)
        t2 = Tensor([[4], [5], [6]], requires_grad = True)

        t3 = t1 @ t2
        t3.backward(Tensor([[1]]))

        print(t1.grad.data.tolist())
        print(t2.grad.data.tolist())

TestTensorMul().test_simple_matmul()
