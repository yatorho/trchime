import trchime as tce

class test_conv_filter:

    def test_shape(self):
        img = tce.random.randn(32, 4, 27, 27)

        fter = tce.random.randn(10, 4, 3, 3, requires_grad = True)

        bias = tce.random.randn(10, 1, requires_grad = True)

        out = tce.nn.Conv2D(img, fter, bias, stride = 2, pad = 1)
        out.sum().backward()
        print(out.shape)
        print(bias.grad.shape)

    def test_forward(self):
        img = tce.arange(98).reshape((2, 1, 7, 7))
        print('img:\n', img.data)
        fter = tce.ones((3, 1, 3, 3))
        print('filter:\n', fter.data)
        bias = tce.Tensor([[1], [2], [3]])
        print('bias:\n', bias.data)

        out = tce.nn.Conv2D(img, fter, bias)
        print('res:\n', out.data)

    def test_backward(self):
        img = tce.arange(36).reshape((3, 2, 2, 3))
        print('img:\n', img.data)
        fter = tce.Tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype = tce.float64, requires_grad = True).reshape((1, 2, 2, 2))
        print('filter:\n', fter.data)
        bias = tce.Tensor([[1]], requires_grad = True)
        print('bias:\n', bias.data)

        out = tce.nn.Conv2D(img, fter, bias, stride = 1)
        print('res:\n', out.data)

        out.sum().backward()
        print('filter\'s gradient:\n', fter.grad.data)
        print('bias\'s gradient:\n', bias.grad.data)

    def padd_backward(self):
        img = tce.random.randn(32, 4, 26, 26, requires_grad = True)

        fter = tce.random.randn(10, 4, 3, 3, requires_grad = True)

        bias = tce.random.randn(10, 1)

        out = tce.nn.Conv2D(img, fter, bias, stride = 5, pad = 1)
        out.sum().backward()
        print(out.shape)
        print(fter.grad.shape)
        print(img.grad.shape)



# test_conv_filter().test_shape()
# test_conv_filter().test_backward()
test_conv_filter().test_forward()
# test_conv_filter().padd_backward()
