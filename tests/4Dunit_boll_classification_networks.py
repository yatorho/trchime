import trchime as tce
import numpy as np


np.random.randn()

def getSource():
    """
    collect your datasets
    :return:
    """
    # x_train 20000, 4
    x_train = 2 * (tce.random.random((20000, 4)) - 0.5)
    """
    in: 0, 1
    out: 1, 0
    """
    y_train = tce.zeros((20000, 2))

    for i in range(20000):
        # assign value to t_train
        if (x_train[i, 0] **2 + x_train[i, 1]**2 + x_train[i, 2]**2 + x_train[i, 3]**2 < 1).data:
            y_train[i, 0] = 0
            y_train[i, 1] = 1
        else:
            y_train[i, 0] = 1
            y_train[i, 1] = 0

    return x_train, y_train

def gettest():
    """
    collect your datasets
    :return:
    """
    # x_train 20000, 4
    x_train = 2 * (tce.random.random((5000, 4)) - 0.5)
    """
    in: 0, 1
    out: 1, 0
    """
    y_train = tce.zeros((5000, 2))

    for i in range(5000):
        # assign value to t_train
        if (x_train[i, 0] **2 + x_train[i, 1]**2 + x_train[i, 2]**2 + x_train[i, 3]**2 < 1).data:
            y_train[i, 0] = 0
            y_train[i, 1] = 1
        else:
            y_train[i, 0] = 1
            y_train[i, 1] = 0

    return x_train, y_train

x_data, y_data = getSource()
x_test, y_test = gettest()

"""
(20000, 4) 

1layer netuals' nums weight (4, 50) bias: (1, 50)
2layer netuals' nums (50, 30)       (1, 30)
3layer netuals' nums (30, 2)        (1, 2)


z^l = w^l@a^l-1 + b^l
a^l = func(z^l)

"""
w1 = 0.1 *tce.random.randn(4, 50, requires_grad = True)
b1 = 0.1 *tce.random.randn(1, 50, requires_grad = True)

w2 = 0.1 *tce.random.randn(50, 30, requires_grad = True)
b2 = 0.1 *tce.random.randn(1, 30, requires_grad = True)

w3 = 0.1 *tce.random.randn(30, 2, requires_grad = True)
b3 = 0.1 *tce.random.randn(1, 2, requires_grad = True)

def predict(inputs):
    z1 = inputs @ w1 + b1
    a1 = tce.ReLU(z1)

    z2 = a1 @ w2 + b2
    a2 = tce.sigmoid(z2)

    z3 = a2 @ w3 + b3
    return tce.sigmoid(z3)


lr = 0.5
epoch = 20
batch_size = 32  # 2**n

for epoch in range(epoch):
    epoch_loss = 0
    predicted_test = predict(x_test)  # 5000, 2

    arg_pre_test = tce.argmax(predicted_test, axis = 1)  # 5000  0, 1, 0 , 1...
    actual_test = tce.argmax(y_test, axis = 1)  # 5000  1, 1, 0, 1...

    rate = tce.mean(1 * (actual_test == arg_pre_test))

    for start in range(0, 20000, batch_size):
        """
        the minibatch in sgd
        
        """
        x_inputs = x_data[start: start + batch_size]

        acctual = y_data[start: start + batch_size]

        predicted = predict(x_inputs)

        loss = tce.sum((predicted - acctual)**2, axis = 1, keepdims = True)  # 32, 1

        loss = tce.mean(loss)

        loss.backward()

        w1.assign_sub(lr * w1.grad)
        w2.assign_sub(lr * w2.grad)
        w3.assign_sub(lr * w3.grad)

        b1.assign_sub(lr * b1.grad)
        b2.assign_sub(lr * b2.grad)
        b3.assign_sub(lr * b3.grad)

        epoch_loss += loss

    predicted_test = predict(x_test)  # 5000, 2

    arg_pre_test = tce.argmax(predicted_test, axis = 1)  # 5000  0, 1, 0 , 1...
    actual_test = tce.argmax(y_test, axis = 1)  # 5000  1, 1, 0, 1...

    rate = tce.mean(1*(actual_test == arg_pre_test))




    print('epoch:', epoch, 'loss:', epoch_loss.data, 'acc:', rate.data)
