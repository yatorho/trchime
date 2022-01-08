"""
The idea here is that we'd like to use our library
to minimize a function, say (x - c) ** 2
"""
import trchime as tce

# declare variable x and constant c.
x = tce.Variable([10., -10., -5., 6., 3., 1.])
c = tce.Constant([1., 2., -7., 6., tce.pi, -tce.e])


# minimize (x - c) ** 2 by 30 iterations
for i in range(30):

    y = (x - c) ** 2
    z = y.sum()
    z.backward()  # call backward method to compute x's gradient

    # update x with method `assign_sub`
    # specially, updating variable with `-=` operator would be inefficiently and slowly horribly
    # we strongly recomend this way to update your network's parameters instead of `-=` operator
    x.assign_sub(0.4 * x.grad.data)

    print('z:', z.data)

# show x's value
print('\nvalue of x:\n', x.data)
