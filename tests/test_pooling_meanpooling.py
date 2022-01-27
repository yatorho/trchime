import trchime as tce

class test_pooling_meanpooling:

    def test_shape(self):
        img = tce.random.random((12, 3, 32, 28))
        print('img:\n', img.shape)

        out = tce.nn.Meanpool2D(img)
        print('out:\n', out.shape)

    def test_forward(self):
        img = tce.trange(1, 65).reshape((2, 2, 4, 4))
        print('img:\n', img.data)

        out = tce.nn.Meanpool2D(img, stride = 1)
        print('out:\n', out.data)

    def test_backward(self):
        img = tce.trange(0, 64, requires_grad = True).reshape((2, 2, 4, 4))
        print('img:\n', img.data)

        out = tce.nn.Meanpool2D(img, stride = 2)
        print('out:\n', out.data)

        grad = tce.ones((2, 2, 2, 2))
        grad[1, 0, 0, 1] = 3
        out.backward(grad)

        print('grad:\n', img.grad.data)


t = test_pooling_meanpooling()
# t.test_shape()
# t.test_forward()
t.test_backward()
