import trchime as tce

def get_dataset(p_sum):
    # randomly generate points
    x_data = tce.random.randn(p_sum, 3)
    # set linear problem's coefficient
    coef = tce.Constant([[-1], [3], [-2]])

    # construct standard module: y = kx + b
    y_data = x_data @ coef + 5
    # set noise for module
    y_data += tce.random.randn(p_sum, 1)

    return x_data, y_data

class linearModel(tce.Module):
    def __init__(self):
        super().__init__()
        # declare coefficient variable w with initialize randomly
        self.w = tce.random.randn(3, 1, requires_grad = True)
        # declare bias with initialize randomly
        self.b = tce.random.randn(requires_grad = True)

    def predict(self, inputs):
        # define your forward here
        return inputs @ self.w + self.b

# collect training data
points_sum = 200
x, y = get_dataset(points_sum)

# instantiate your model
model = linearModel()

# compile your model
model.compile(optimizer = tce.nn.SGD_OPTIMIZER,  # choose stochastic gradient descent optimizer
              loss = tce.nn.MSELOSS,  # set mean square loss function
              learning_rate = 0.01)  # set learning rate

# train your model
model.fit(x, y,  # input training data
          batch_size = 32,  # set batch_size and epochs
          epochs = 100)

# show parameters:
print(f"w: \n{model.w.data}\n")
print(f"b: \n{model.b.data}")
