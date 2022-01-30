from typing import List

import trchime as tce
from trchime import Module, Tensor
from trchime.nn import Conv2D, Maxpool2D, LOSS
import numpy as np
import cv2
# from vgg.imagenet_classes import class_names

class VGG(Module):
    def __init__(self):
        super().__init__()
        self.parameters_list: List[tce.Tensor] = []

        self.kernel1_1 = tce.zeros((64, 3, 3, 3), dtype = tce.float32)
        self.bias1_1 = tce.zeros((64, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel1_1, self.bias1_1]

        self.kernel1_2 = tce.zeros((64, 64, 3, 3), dtype = tce.float32)
        self.bias1_2 = tce.zeros((64, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel1_2, self.bias1_2]

        self.kernel2_1 = tce.zeros((128, 64, 3, 3), dtype = tce.float32)
        self.bias2_1 = tce.zeros((128, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel2_1, self.bias2_1]

        self.kernel2_2 = tce.zeros((128, 128, 3, 3), dtype = tce.float32)
        self.bias2_2 = tce.zeros((128, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel2_2, self.bias2_2]

        self.kernel3_1 = tce.zeros((256, 128, 3, 3), dtype = tce.float32)
        self.bias3_1 = tce.zeros((256, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel3_1, self.bias3_1]

        self.kernel3_2 = tce.zeros((256, 256, 3, 3), dtype = tce.float32)
        self.bias3_2 = tce.zeros((256, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel3_2, self.bias3_2]

        self.kernel3_3 = tce.zeros((256, 256, 3, 3), dtype = tce.float32)
        self.bias3_3 = tce.zeros((256, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel3_3, self.bias3_3]

        self.kernel4_1 = tce.zeros((512, 256, 3, 3), dtype = tce.float32)
        self.bias4_1 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel4_1, self.bias4_1]

        self.kernel4_2 = tce.zeros((512, 512, 3, 3), dtype = tce.float32)
        self.bias4_2 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel4_2, self.bias4_2]

        self.kernel4_3 = tce.zeros((512, 512, 3, 3), dtype = tce.float32)
        self.bias4_3 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel4_3, self.bias4_3]

        self.kernel5_1 = tce.zeros((512, 512, 3, 3), dtype = tce.float32)
        self.bias5_1 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel5_1, self.bias5_1]

        self.kernel5_2 = tce.zeros((512, 512, 3, 3), dtype = tce.float32)
        self.bias5_2 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel5_2, self.bias5_2]

        self.kernel5_3 = tce.zeros((512, 512, 3, 3), dtype = tce.float32)
        self.bias5_3 = tce.zeros((512, 1), dtype = tce.float32)
        self.parameters_list += [self.kernel5_3, self.bias5_3]

        self.fc1w = tce.zeros((25088, 4096), dtype = tce.float32)
        self.fc1b = tce.zeros((1, 4096), dtype = tce.float32)
        self.parameters_list += [self.fc1w, self.fc1b]

        self.fc2w = tce.zeros((4096, 4096), dtype = tce.float32)
        self.fc2b = tce.zeros((1, 4096), dtype = tce.float32)
        self.parameters_list += [self.fc2w, self.fc2b]

        self.fc3w = tce.zeros((4096, 1000), dtype = tce.float32)
        self.fc3b = tce.zeros((1, 1000), dtype = tce.float32)
        self.parameters_list += [self.fc3w, self.fc3b]


    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())

        for i, k in enumerate(keys):
            # weight = weights[k]
            if i % 2 == 0:
                if i < 26:
                    weight = weights[k].transpose(3, 2, 0, 1)
                else:
                    weight = weights[k]
            else:
                if i >= 26:
                    weight = weights[k].reshape((1, -1))
                else:
                    weight = weights[k].reshape((-1, 1))
            print(i, k, weight.shape)
            # self.parameters_list[i].data = weight
            self.parameters_list[i].assign(weight)

    def predict(self, inputs):  # (1, 3, 224, 224)

        # inputs= cv2.resize(inputs, (224, 224))
        # inputs = inputs.transpose(2, 0, 1).reshape((1, 3, 224, 224))
        # inputs = tce.Tensor(inputs)
        mean = tce.Constant([123.68, 116.779, 103.939]).reshape((1, 3, 1, 1)).astype(tce.float32)
        inputs -= mean  # (1, 3, 224, 224)

        self.conv1_1 = Conv2D(inputs, self.kernel1_1, self.bias1_1, pad = 1)  # (1, 64, 224, 224)
        self.conv1_1 = tce.ReLU(self.conv1_1)

        conv1_2 = Conv2D(self.conv1_1, self.kernel1_2, self.bias1_2, pad = 1)  # (1, 64, 224, 224)
        conv1_2 = tce.ReLU(conv1_2)

        pool1 = Maxpool2D(conv1_2, f_shape = (2, 2))  # (1, 64, 112, 112)

        self.conv2_1 = Conv2D(pool1, self.kernel2_1, self.bias2_1, pad = 1)  # (1, 128, 112, 112)
        self.conv2_1 = tce.ReLU(self.conv2_1)

        conv2_2 = Conv2D(self.conv2_1, self.kernel2_2, self.bias2_2, pad = 1)  # (1, 128, 112, 112)
        conv2_2 = tce.ReLU(conv2_2)

        pool2 = Maxpool2D(conv2_2, f_shape = (2, 2))  # （1， 128， 56， 56）

        self.conv3_1 = Conv2D(pool2, self.kernel3_1, self.bias3_1, pad = 1)  # (1, 256, 56, 56)
        self.conv3_1 = tce.ReLU(self.conv3_1)

        conv3_2 = Conv2D(self.conv3_1, self.kernel3_2, self.bias3_2, pad = 1)  # (1, 256, 56, 56)
        conv3_2 = tce.ReLU(conv3_2)

        conv3_3 = Conv2D(conv3_2, self.kernel3_3, self.bias3_3, pad = 1)  # (1, 256, 56, 56)
        conv3_3 = tce.ReLU(conv3_3)

        pool3 = Maxpool2D(conv3_3, f_shape = (2, 2))  # (1, 256, 28, 28)

        self.conv4_1 = Conv2D(pool3, self.kernel4_1, self.bias4_1, pad = 1)  # (1, 512, 28, 28)
        self.conv4_1 = tce.ReLU(self.conv4_1)

        self.conv4_2 = Conv2D(self.conv4_1, self.kernel4_2, self.bias4_2, pad = 1)  # (1, 512, 28, 28)
        self.conv4_2 = tce.ReLU(self.conv4_2)

        conv4_3 = Conv2D(self.conv4_2, self.kernel4_3, self.bias4_3, pad = 1)  # (1, 512, 28, 28)
        conv4_3 = tce.ReLU(conv4_3)

        pool4 = Maxpool2D(conv4_3, f_shape = (2, 2))  # (1, 512, 14, 14)

        self.conv5_1 = Conv2D(pool4, self.kernel5_1, self.bias5_1, pad = 1)  # (1, 512, 14, 14)
        self.conv5_1 = tce.ReLU(self.conv5_1)

        conv5_2 = Conv2D(self.conv5_1, self.kernel5_2, self.bias5_2, pad = 1)  # (1, 512, 14, 14)
        conv5_2 = tce.ReLU(conv5_2)  # (1, 512, 14, 14)

        conv5_3 = Conv2D(conv5_2, self.kernel5_3, self.bias5_3, pad = 1)  # (1, 512, 14, 14)
        conv5_3 = tce.ReLU(conv5_3)

        pool5 = Maxpool2D(conv5_3, f_shape = (2, 2))  # (1, 512, 7, 7)

        pool5 = pool5.transpose(0, 2, 3, 1).reshape((1, 512 * 7 * 7))  # (1, 25088)

        fc1 = pool5 @ self.fc1w + self.fc1b  # (1, 4096)
        fc1 = tce.ReLU(fc1)

        fc2 = fc1 @ self.fc2w + self.fc2b  # (1, 4096)
        fc2 = tce.ReLU(fc2)

        fc3 = fc2 @ self.fc3w + self.fc3b  # (1, 1000)
        return tce.softmax(fc3, axis = 1)


def compute_content_cost(a_C: Tensor, a_G: Tensor):
    m, C, H, W = a_C.shape

    a_C_unrolled = a_C.reshape((C, H *W))
    a_G_unrolled = a_G.reshape((C, H *W))

    J_content = tce.sum((a_C_unrolled - a_G_unrolled) ** 2) / (4 * H * W * C)

    return J_content

def gram_matrix(A: Tensor):  # A shape: (c, h * w)
    GA = A @ A.transpose(1, 0)

    return GA  # (c, c)

def compute_layer_style_const(a_S: Tensor, a_G: Tensor):
    m, C, H, W = a_S.shape

    a_S = a_S.reshape((C, H * W))
    a_G = a_G.reshape((C, H * W))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tce.sum((GS - GG) ** 2) / (4 * (C **2) * (H * W) ** 2)

    return J_style_layer


def compute_style_cost(model: 'NST', STYLE_LAYERS):
    J_style = 0
    a_S_list = []
    a_G_list = []

    model.net.predict(model.style)
    for layer_name, coef in STYLE_LAYERS:
        a_S = model.net[layer_name]
        a_S_list.append(a_S)

    model.net.predict(model.picture)
    for layer_name, coef in STYLE_LAYERS:
        a_G = model.net[layer_name]
        a_G_list.append(a_G)

    for i, a in enumerate(a_S_list):
        J_style_layer = compute_layer_style_const(a, a_G_list[i])
        J_style += STYLE_LAYERS[i][1] * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha=10.0, beta=40.0):
    return alpha * J_content + beta * J_style

class NST(Module):

    def __init__(self, content_file: Tensor, style_file: Tensor, p_size: tuple = None):
        super().__init__()
        self.content = content_file  # (1, 3, h, w)
        self.content.non_gradient()

        self.style = style_file  # (1, 3, h, w)
        self.style.non_gradient()

        if p_size is None:
            p_size = self.content.shape
        self.p_size = p_size

        self.picture = tce.random.random(self.p_size, requires_grad = True) * 256

    def load_net(self, model: Module):
        self.net = model

    def remove_net(self):
        self.net = None

    def train(self, inputs):
        return None

    def predict(self, inputs):
        m, C, H, W = self.picture.shape
        # mean = tce.Constant([123.68, 116.779, 103.939]).reshape((1, 3, 1, 1)).astype(tce.float32)
        out = self.picture
        out = out.reshape((C, H, W))
        out = out.transpose(1, 2, 0)

        return out.data

class nst_Loss(LOSS):

    def __init__(self):
        super().__init__('nst_loss')

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', model: 'NST' = None):

        model.net.predict(model.content)
        a_C = model.net['conv4_2']

        model.net.predict(model.picture)
        a_G = model.net['conv4_2']

        J_content = compute_content_cost(a_C, a_G)

        J_style = compute_style_cost(model, STYLE_LAYERS)

        J = total_cost(J_content, J_style)
        self.loss = J

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


vgg = VGG()
vgg.load_weights('../vgg/vgg16_weights.npz')

content = cv2.imread('../vgg/stars.png')
content = cv2.resize(content, (224, 224))
content = content.transpose(2, 0, 1).reshape((1, 3, 224, 224))
content = Tensor(content)

style = cv2.imread('../vgg/laska.png')
style = cv2.resize(style, (224, 224))
style = style.transpose(2, 0, 1).reshape((1, 3, 224, 224))
style = Tensor(style)

nstnn = NST(content, style)
nstnn.load_net(vgg)

nstloss = nst_Loss()

nstnn.compile(optimizer = tce.nn.ADAM_OPTIMIZER,
              loss = nstloss,
              learning_rate = 10)

nstnn.fit(epochs = 100,
          valid_inputs = True)

picture = nstnn.predict(None)

nstnn.remove_net()
nstnn.summary()

cv2.imshow('p', picture)
cv2.waitKey(0)
cv2.imwrite('generate.png', picture)

tce.savemodel(nstnn, 'nst.pkl')
