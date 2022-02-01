import trchime as tce

class test_conv_img:

    def test_shape(self):
        img = tce.random.randn(32, 4, 27, 27, requires_grad = True)

        fter = tce.random.randn(10, 4, 3, 3, requires_grad = True)

        bias = tce.random.randn(10, 1)

        out = tce.nn.Conv2D(img, fter, bias, stride = 2, pad = 1)
        out.sum().backward()
        print(out.shape)
        print(img.grad.shape)

    def test_forward(self):
        img = tce.arange(24).reshape((2, 2, 2, 3))
        print('img:\n', img.data)
        fter = tce.Tensor([[[1]], [[1]]]).reshape((1, 2, 1, 1))
        print('filter:\n', fter.data)
        bias = tce.Tensor([[0]])
        print('bias:\n', bias.data)

        out = tce.nn.Conv2D(img, fter, bias)
        print('res:\n', out.data)

    def test_backward(self):
        img = tce.trange(32, requires_grad = True).reshape((2, 2, 2, 4))
        print('img:\n', img.data)
        fter = tce.Tensor([[[0, 1], [1, 1]], [[0, 1], [0, 1]]], dtype = tce.float64).reshape((1, 2, 2, 2))
        print('filter:\n', fter.data)
        bias = tce.Tensor([[1]])
        print('bias:\n', bias.data)

        out = tce.nn.Conv2D(img, fter, bias, stride = 2, pad = 1)
        print('res:\n', out.data)

        out.sum().backward()
        print('img\'s gradient:\n', img.grad.data)

    def padd_backward(self):
        img = tce.random.randn(32, 4, 26, 26, requires_grad = True)

        fter = tce.random.randn(10, 4, 3, 3, requires_grad = True)

        bias = tce.random.randn(10, 1)

        out = tce.nn.Conv2D(img, fter, bias, stride = 5, pad = 1)
        out.sum().backward()  # out 1, 1, 2, 3
        print(out.shape)
        print(fter.grad.shape)
        print(img.grad.shape)

t = test_conv_img()

# t.test_shape()
# t.test_forward()  # 1, 1, 2, 1
t.test_backward()




