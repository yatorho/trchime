import trchime as tce

# m = tce.nn.Layer_Manager()
#
# m.add(tce.nn.Convolution_layer2D((3, 3), 3))
# m.add(tce.nn.MaxPooling_layer2D((2, 2),  tce.nn.Activation.RELUX_ACTIVATION))
# m.add(tce.nn.Convolution_layer2D((4, 4), 12))
# m.add(tce.nn.AveragePool_layer2D((2, 2),  tce.nn.Activation.RELUX_ACTIVATION))
# m.add(tce.nn.Flatten())
# m.add(tce.nn.Dense(45, tce.nn.Activation.ELU_ACTIVATION))
# m.add(tce.nn.Dense(10, tce.nn.Activation.SOFTMAX_ACTIVATION))
#
# m.construct_grap((1, 1, 28, 28))
#
# print(m.forward(tce.random.randn(1, 1, 28, 28)).shape)

model = tce.Module()
model.add(tce.nn.Convolution_layer2D((3, 3), 3))
model.add(tce.nn.MaxPooling_layer2D((2, 2),  tce.nn.Activation.RELUX_ACTIVATION))
model.add(tce.nn.Convolution_layer2D((4, 4), 12))
model.add(tce.nn.AveragePool_layer2D((2, 2),  tce.nn.Activation.RELUX_ACTIVATION))
model.add(tce.nn.Flatten())
model.add(tce.nn.Dense(45, tce.nn.Activation.ELU_ACTIVATION))
model.add(tce.nn.Dense(10, tce.nn.Activation.SOFTMAX_ACTIVATION))

model.compile(optimizer = tce.nn.ADAM_OPTIMIZER, loss = tce.nn.CATEGORYLOSS, learning_rate = 0.1)
model.fit(tce.random.randn(100, 1, 28, 28), tce.random.randint(0, 10, (100, 10)), batch_size = 32, epochs = 100)
# model.layer_manager.construct_grap((1, 4, 28, 28))
# for t in model.parameters():
#     print(t.shape)
