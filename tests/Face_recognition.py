import numpy as np
import trchime as tce
from PIL import Image
import matplotlib.pyplot as plt
import cv2

#     a = np.random.random_integers(low=10, high=100, size=[5, 4, 3])
#     bs.append(a[np.newaxis, :])
# c = np.concatenate(bs)
# print(bs.__len__())
#

picture_list = list()
txt_list = list()

f1 = open('C:\\Users\\lenovo\\Desktop\\real_face\\face\\msg.txt', 'r')
for i in range(0, 2100):
    temp_str = str(i + 1)
    temp_str = temp_str.zfill(4)
    filename = 'C:\\Users\\lenovo\\Desktop\\real_face\\face\\face_all\\' + temp_str + '.png'
    img_cv = cv2.imread(filename)
    img_cv = img_cv.transpose((2, 0, 1))
    img_cv = img_cv.reshape((3, 128, 128))
    picture_list.append(img_cv[np.newaxis, :])

face_input = np.concatenate(picture_list)

for i in range(2100):
    str1 = f1.readline()
    str1 = str1.split('\n')[0]
    list1 = str1.split(' ')
    del list1[0]
    for j, v in enumerate(list1):
        list1[j] = float(v)
    temp = np.asarray(list1)
    txt_list.append(temp[np.newaxis, :])

face_output = np.concatenate(txt_list)

face_input = face_input.reshape((2100, 3, 128, 128))

face_input = tce.Tensor(face_input, requires_grad=True)
face_output = tce.Tensor(face_output, requires_grad=True)
print(face_input.shape)
print(face_output.shape)


# print(face_output)

class Face_Recognition(tce.Module):
    def __init__(self):
        self.init()
        self.output2 = 0
        self.filter1 = tce.random.randn(8, 3, 5, 5, requires_grad=True) * 0.05  #
        self.bias1 = tce.random.randn(8, 1, requires_grad=True) * 0.05
        self.filter2 = tce.random.randn(16, 8, 7, 7, requires_grad=True) * 0.05
        self.bias2 = tce.random.randn(16, 1, requires_grad=True) * 0.05
        self.filter3 = tce.random.randn(32, 16, 5, 5, requires_grad=True) * 0.05
        self.bias3 = tce.random.randn(32, 1, requires_grad=True) * 0.05
        self.filter4 = tce.random.randn(32, 32, 3, 3, requires_grad=True) * 0.05
        self.bias4 = tce.random.randn(32, 1, requires_grad=True) * 0.05

        self.w1 = tce.random.randn(1568, 64, requires_grad=True) * 0.05
        self.b1 = tce.random.randn(1, 64, requires_grad=True) * 0.05
        self.w2 = tce.random.randn(64, 32, requires_grad=True) * 0.05
        self.b2 = tce.random.randn(1, 32, requires_grad=True) * 0.05
        self.w3 = tce.random.randn(32, 2, requires_grad=True) * 0.05
        self.b3 = tce.random.randn(1, 2, requires_grad=True) * 0.05
        self.w4 = tce.random.randn(32, 6, requires_grad=True) * 0.05
        self.b4 = tce.random.randn(1, 6, requires_grad=True) * 0.05

    def predict(self, inputs):
        # inputs 1, 3, 128, 128
        c1 = tce.nn.Conv2D(inputs, self.filter1, self.bias1)  # 1, 8, 124, 124
        p1 = tce.nn.Maxpool2D(c1, (2, 2), stride=2)  # 1, 8, 68, 68
        p1 = tce.ReLU(p1)

        c2 = tce.nn.Conv2D(p1, self.filter2, self.bias2)  # 1, 16, 56, 56
        p2 = tce.nn.Maxpool2D(c2, (2, 2), stride=2)  # 1, 16, 28, 28
        p2 = tce.ReLU(p2)

        c3 = tce.nn.Conv2D(p2, self.filter3, self.bias3, pad=2)  # 1, 32, 28, 28
        p3 = tce.nn.Maxpool2D(c3, (2, 2), stride=2)  # 1, 32, 14, 14
        p3 = tce.ReLU(p3)

        c4 = tce.nn.Conv2D(p3, self.filter4, self.bias4, pad=1)  # 1, 32, 14, 14
        p4 = tce.nn.Maxpool2D(c4, (2, 2), stride=2)  # 1, 32, 7, 7
        p4 = tce.ReLU(p4)

        # 60000, 300
        z1 = p4.reshape((-1, 1568)) @ self.w1 + self.b1  # 1, 64
        a1 = tce.ReLU(z1)

        z2 = a1 @ self.w2 + self.b2  # 1, 32
        a2 = tce.ReLU(z2)

        # z3 = a2 @ self.w3 + self.b3  # 1, 2
        # output1 = tce.sigmoid(z3)

        z4 = a2 @ self.w4 + self.b4  # 1, 4
        self.output2 = tce.sigmoid(z4)

        return self.output2

    def showlayer1(self):
        pass


class MYLOSS(tce.nn.LOSS):

    def __init__(self):
        super().__init__()

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', model: 'Module' = None) -> None:
        copy = np.zeros(actual.shape)
        copy[:, 1:] = actual[:, 0:1].data
        copy[:, 0:2] = 1

        self.loss = (((predicted*copy)-actual)**2).sum()


face = Face_Recognition()
myloss = MYLOSS()
face.compile(optimizer=tce.nn.ADAM_OPTIMIZER, loss=myloss, learning_rate=0.1)
9
# model = tce.loadmodel('model/mnist_conv.pkl')
face.fit(face_input, face_output, batch_size=35, epochs=10, show_acc_tr=True, show_acc=True)
