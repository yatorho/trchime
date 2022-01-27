import trchime as tce
import numpy as np

a = np.array([[1, 2, 3],
              [2, 3, 4],
              [-2, 4, 1]])
a = tce.Variable(a)  # declare a variable

b = np.array([2, 3, 4])
b = tce.Variable(b)  # declare another variable

y = a * b  # compute value of a * b
z = y.sum()  # get sum of y

# it's easy to compute a, b's gradient with calling backward method of z
z.backward()

# show the results:
print(f"value of y:\n{y.data}\n")
print(f"sum of y:\n{z.data}\n")
print(f"gradient of a:\n{a.grad.data}\n")
print(f"gradient of b:\n{b.grad.data}\n")
