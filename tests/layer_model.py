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

model = tce.Module()
model.add(tce.nn.Dense(1))
# model.layer_manager.construct_grap((100, 3))

points_sum = 200
x, y = get_dataset(points_sum)

model.compile(optimizer = tce.nn.SGD_OPTIMIZER,
              loss = tce.nn.MSELOSS,
              learning_rate = 0.1)

model.fit(x, y,
          batch_size = 32,
          epochs = 100)

print(f"w: \n{model.layer_manager.layers_list[0].weight.data}\n")
print(f"b: \n{model.layer_manager.layers_list[0].bias.data}")
