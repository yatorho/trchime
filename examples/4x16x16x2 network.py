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

# construct your model by inherit class `tce.Module`
# you need to overwrite two methods
class unit_Model(tce.Module):
    def __init__(self):
        # initialize paramters' matrix randomly
        self.w1 = tce.random.randn(4, 16, requires_grad = True) * 0.1
        self.b1 = tce.random.randn(1, 16, requires_grad = True) * 0.1

        self.w2 = tce.random.randn(16, 16, requires_grad = True) * 0.1
        self.b2 = tce.random.randn(1, 16, requires_grad = True) * 0.1

        self.w3 = tce.random.randn(16, 2, requires_grad = True) * 0.1
        self.b3 = tce.random.randn(1, 2, requires_grad = True) * 0.1

    def predict(self, inputs):
        # define your forward model here
        # 4x16x16x2
        z1 = inputs @ self.w1 + self.b1
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = tce.ReLU(z2)

        z3 = a2 @ self.w3 + self.b3
        return tce.sigmoid(z3)  # output layer use sigmoid activate function

# collect dataset
points_sum = 1000
x, y = get_dataset(points_sum)

# instantiate your model
model = unit_Model()

# compile your model
model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,  # choose stochastic gradient descent optimizer
              loss = tce.nn.MSELOSS,  # set mean square loss function
              learning_rate = 0.1)  # set learning rate

# train your model
model.fit(x, y,  # input training data
          batch_size = 32,  # set batch_size and epochs
          epochs = 100,
          validation_split = 0.2,  # split 20% of trainingset as testingset
          show_acc = True)  # show accuray per epoch
