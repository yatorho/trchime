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


model = tce.Module()

model.add(tce.nn.Dense(32, activation = tce.nn.Activation.NONE))
model.add(tce.nn.Batch_normalize_layer(tce.nn.Activation.RELU_ACTIVATION))
model.add(tce.nn.Dense(32, activation = tce.nn.Activation.NONE))
model.add(tce.nn.Batch_normalize_layer(tce.nn.Activation.RELU_ACTIVATION))
model.add(tce.nn.Dense(2, activation = tce.nn.Activation.NONE))
model.add(tce.nn.Batch_normalize_layer(tce.nn.Activation.SIGMOID_ACTIVATION))


model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,
              loss = tce.nn.MSELOSS,
              learning_rate = 1)

model.fit(x_data,
          y_data,
          batch_size = 32,
          epochs = 100,
          validation_split = 0.2,
          show_acc_tr = True)

