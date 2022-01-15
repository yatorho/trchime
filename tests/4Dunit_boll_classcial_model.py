import trchime as tce


def getSource():
    """
    collect your datasets
    :return:
    """
    # x_train 20000, 4
    x_train = 2 * (tce.random.random((2000, 4)) - 0.5)
    """
    in: 0, 1
    out: 1, 0
    """
    y_train = tce.zeros((2000, 2))

    for i in range(2000):
        # assign value to t_train
        if (x_train[i, 0] **2 + x_train[i, 1]**2 + x_train[i, 2]**2 + x_train[i, 3]**2 < 1).data:
            y_train[i, 0] = 0
            y_train[i, 1] = 1
        else:
            y_train[i, 0] = 1
            y_train[i, 1] = 0

    return x_train, y_train

x_data, y_data = getSource()


class _4dunit_bool_Model(tce.Module):

    def __init__(self):
        self.w1 = 0.1 * tce.random.randn(4, 16, requires_grad = True)
        self.b1 = 0.1 * tce.random.randn(1, 16, requires_grad = True)

        self.w2 = 0.1 * tce.random.randn(16, 16, requires_grad = True)
        self.b2 = 0.1 * tce.random.randn(1, 16, requires_grad = True)

        self.w3 = 0.1 * tce.random.randn(16, 2, requires_grad = True)
        self.b3 = 0.1 * tce.random.randn(1, 2, requires_grad = True)

    def predict(self, inputs):
        z1 = inputs @ self.w1 + self.b1  # 32 * 100
        # z1 = tce.nn.dropout_layer(z1, 0.8, 1, True)
        a1 = tce.ReLU(z1)


        z2 = a1 @ self.w2 + self.b2
        # z2 = tce.nn.dropout_layer(z2, 0.8, 1, True)
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3 + self.b3
        return tce.sigmoid(z3)

model = _4dunit_bool_Model()

model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,
              loss = tce.nn.MSELOSS,
              learning_rate = 0.1)

model.fit(x_data,
          y_data,
          batch_size = 32,
          epochs = 100,
          validation_split = 0.2,
          show_acc_tr = True)

