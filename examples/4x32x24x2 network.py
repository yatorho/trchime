import trchime as tce
import os

def get_dataset(p_sum):
    """
    Produce dataset for 2 classification problem.
    Generate some points in 4-dimensional space and distribute them
    to 2 labels.
    if points are in unit boll of 4-dimension space, they would
    be marked as label 1.
    If not, they would be marked as label 2
    """
    x_train = 5 * (tce.random.random((p_sum, 4)) - 0.5)
    y_train = tce.zeros((p_sum, 2))

    for i in range(p_sum):
        if (tce.sum(x_train[i, :] ** 2) < 1).data:  # in -> labels: [0, 1]
            y_train[i, 0] = 0
            y_train[i, 1] = 1
        else:  # out -> labels: [1, 0]
            y_train[i, 0] = 1
            y_train[i, 1] = 0

    return x_train, y_train

# collect dataset
points_sum = 2000
x, y = get_dataset(points_sum)

model_save_path ='4x32x24x2network.pkl'

if os.path.exists(model_save_path):
    model = tce.loadmodel(model_save_path)
else:
    # instantiate your model
    model = tce.Module()

    # add some computing layer for your model
    model.add(tce.nn.Dense(nums = 32,
                           activation = tce.nn.Activation.RELU_ACTIVATION))
    model.add(tce.nn.Dense(nums = 24,
                           activation = tce.nn.Activation.RELU_ACTIVATION))
    model.add(tce.nn.Dense(nums = 2,
                           activation = tce.nn.Activation.SIGMOID_ACTIVATION))

# compile your model
model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,
              loss = tce.nn.MSELOSS)

# train your model
model.fit(x, y,
          batch_size = 32,
          epochs = 50,
          validation_split = 0.3,  # split 20% of trainingsets as testset
          show_acc_tr = True)

tce.savemodel(model, url = '4x32x24x2network.pkl')


