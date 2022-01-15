import trchime as tce

class test_pooling_max:
    def simple_test_shape(self):
        img = tce.random.randn(3, 4, 14, 16, requires_grad = True)
        print('img:', img.data.shape)
        out = tce.nn.Maxpool2D(img, (2, 4), stride = 2, pad = 0)
        print("out:", out.data.shape)
        out.sum().backward()
        print('grad:', img.grad.shape)

    def test_forward(self):
        img = tce.trange(64, requires_grad = True).reshape((2, 2, 4, 4))
        img[1, 0, 2, 1] = 80
        print('img:\n', img.data)
        print(img.shape)
        out = tce.nn.Maxpool2D(img, (2, 2), stride = 1)
        print('out:\n', out.data)
        print(out.shape)

        out.sum().backward()
        # print('grad:\n', img.grad.data)

t = test_pooling_max()
# t.simple_test_shape()
t.test_forward()

