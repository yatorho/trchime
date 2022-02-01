from trchime import core as ad

"""
The idea here is that we'd like to use our library
to minimize a function, say x**2
"""

x = ad.Tensor([10., -10., -5., 6., 3., 1.], requires_grad = True)
c = ad.Tensor([1., 2., 2., 6., 2., 2.])

for i in range(100):
    d = ad.sub(x, c)
    y = ad.sum(ad.mul(d, d))
    y.backward()

    delta_x = ad.mul(ad.Tensor(0.1), x.grad)
    # x = Tensor(x.data - delta_x.data, requires_grad = True)
    x.assign_sub(0.4 * x.grad.data)
    print(y.data)

print(x)
