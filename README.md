# Hello Trchime!

## What's Trchime?

> Tiny deep-learning framework power by python3  and numpy

* [exampls](examples/) --complete code examples of Tchime
* [documentation](docs.md) --trchime development documentaion and usage documentation

## Dependencies

| Name   | Version |
| ------ | ------- |
| Python | 3.7.0+  |
| Numpy  | 1.19.0+ |

## Start

Compute $ y=kx +b $​ gradient with trchime, and ouput the gradient of  $ x $​ .

The whole process is completed by graph calculation and automatic derivation.

``` python	
from trchime import Variable, Constant
k = Constant([[2, 3], [1, 1]])
b = Constant([[7], [3]])
x = Variable([[0], [0]])

y = k @ x + b
z = y.sum()
z.backward()

print(x.grad.data)
```

result

``` python
[[3.],
 [4.]]
```

## What can trchime do?

* Graph Computing
* Auto Gradient
* Algebraic system
* Gradient Descent
* Neural Network API
* Convolutional Neural Network







