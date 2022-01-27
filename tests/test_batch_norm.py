import trchime as tce

def get_dataset(p_sum):
    """
    Produce dataset for 2 classification problem.
    Generate some points in 4-dimensional space and distribute them
    to 2 labels.
    if points are in unit boll of 4-dimension space, they would
    be marked as label 1.
    If not, they would be marked as label 2
    """
    x_train = 2 * (tce.random.random((p_sum, 4)) - 0.5)
    y_train = tce.zeros((p_sum, 2))

    for i in range(p_sum):
        if (tce.sum(x_train[i, :] ** 2) < 1).data:  # in -> labels: [0, 1]
            y_train[i, 0] = 0
            y_train[i, 1] = 1
        else:  # out -> labels: [1, 0]
            y_train[i, 0] = 1
            y_train[i, 1] = 0

    return x_train, y_train

x_data, y_data = get_dataset(500)


class _4dunit_bool_Model(tce.Module):

    def __init__(self):
        super().__init__()
        self.w1 = 0.1 * tce.random.randn(4, 16, requires_grad = True)
        self.gamma1 = 0.1 * tce.random.randn(1, 16, requires_grad = True)
        self.beta1 = 0.1 * tce.random.randn(1, 16, requires_grad = True)

        self.w2 = 0.1 * tce.random.randn(16, 16, requires_grad = True)
        self.gamma2 = 0.1 * tce.random.randn(1, 16, requires_grad = True)
        self.beta2 = 0.1 * tce.random.randn(1, 16, requires_grad = True)

        self.w3 = 0.1 * tce.random.randn(16, 2, requires_grad = True)
        self.gamma3 = 0.1 * tce.random.randn(1, 2, requires_grad = True)
        self.beta3 = 0.1 * tce.random.randn(1, 2, requires_grad = True)

    def predict(self, inputs):
        z1 = inputs @ self.w1  # (32, 16)
        u1 = z1.mean(axis = 0, keepdims = True)  # (1, 16)
        s1 = z1.var(axis = 0, keepdims = True)  # (1, 16)
        z1_norm = (z1 - u1) / tce.sqrt(s1 + 1e-12)  # (32, 16)
        z1 = self.gamma1 * z1_norm + self.beta1  # (32, 16)
        a1 = tce.ReLU(z1)  # (32, 16)


        z2 = a1 @ self.w2
        u2 = z2.mean(axis = 0, keepdims = True)
        s2 = z2.var(axis = 0, keepdims = True)
        z2_norm = (z2 - u2) / tce.sqrt(s2 + 1e-12)
        z2 = self.gamma2 * z2_norm + self.beta2
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3
        u3 = z3.mean(axis = 0, keepdims = True)
        s3 = z3.var(axis = 0, keepdims = True)
        z3_norm = (z3 - u3) / tce.sqrt(s3 + 1e-12)
        z3 = self.gamma3 * z3_norm + self.beta3
        return tce.sigmoid(z3)

model = _4dunit_bool_Model()

model.compile(optimizer = tce.nn.SGD_OPTIMIZER,
              loss = tce.nn.MSELOSS,
              learning_rate = 1)

model.fit(x_data,
          y_data,
          batch_size = 32,
          epochs = 100,
          validation_split = 0.2,
          show_acc_tr = True)

