from trchime.core.tensor import Tensor


class TestTensorSlice:
    def test_simple_slice(self):
        t1 = Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]], requires_grad = True)
        t2 = t1[0:2, :2]
        print(t2)
        t2.backward(Tensor([[2, 2], [-3, -4]]))
        # t1.backward(Tensor([[1, 2, 3], [2, 3, 4]]))


        print(t1.grad.data)
        print(t2.grad.data)

TestTensorSlice().test_simple_slice()
