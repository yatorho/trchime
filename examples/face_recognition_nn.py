import trchime as tce
from trchime import Module, Tensor
from trchime.call import AccuracyBoard
from trchime.nn import Conv2D, Maxpool2D, LOSS
from trchime.datasets import faces_set

import cv2
import os


data_path = 'E:/faces.npz'
faces = faces_set(data_path)  # pass face_set's url
"""
faces dataset consists of 2100 images and their corresponding labels.
The size of the picture is 3 x 128 x 128 (None, C, H, W).
The labels's size is (6,). The first value in the label array indicates whether the image 
corresponding to the serial number has an object to be detected, the second value 
indicates to the probability that the object is a face, 
and the last four data represent the position of the face in the picture(dx, dy, dw, dh).
So, the outputs picture array's shape would be (2100, 3, 128, 128) and
labels array's shape would be (2100, 6)
"""
x_data, y_data = faces.load(whether_tensor = True)  # load data

assert x_data.shape == (2100, 3, 128, 128)
assert y_data.shape == (2100, 6)


class Face_Model(Module):

    def __init__(self):
        self.init()

        # initialize parameter matrix randomly
        # shape of kernels in `Trchime` would be (N, C, H, W)
        # shape of biases in `Trchime` would be (N, 1)
        self.kernel1 = tce.Variable(shape = (8, 3, 5, 5))
        self.bias1 = tce.Variable(shape = (8, 1))

        self.kernel2 = tce.Variable(shape = (16, 8, 7, 7))
        self.bias2 = tce.Variable(shape = (16, 1))

        self.kernel3 = tce.Variable(shape = (32, 16, 5, 5))
        self.bias3 = tce.Variable(shape = (32, 1))

        self.kernel4 = tce.Variable(shape = (32, 32, 3, 3))
        self.bias4 = tce.Variable(shape = (32, 1))

        self.fc1w = tce.Variable(shape = (1568, 256))
        self.fc1b = tce.Variable(shape = (1, 256))

        self.fc2w = tce.Variable(shape = (256, 64))
        self.fc2b = tce.Variable(shape = (1, 64))

        self.fc3w_1 = tce.Variable(shape = (64, 2))
        self.fc3b_1 = tce.Variable(shape = (1, 2))

        self.fc3w_2 = tce.Variable(shape = (64, 4))
        self.fc3b_2 = tce.Variable(shape = (1, 4))

    def predict(self, inputs):  # (?, 3, 128, 128)
        conv1 = Conv2D(inputs, self.kernel1, self.bias1)  # (?, 8, 124, 124)
        conv1 = tce.ReLU(conv1)

        pool1 = Maxpool2D(conv1, f_shape = (2, 2))  # (?, 8, 62, 62)

        conv2 = Conv2D(pool1, self.kernel2, self.bias2)  # (?, 16, 56, 56)
        conv2 = tce.ReLU(conv2)

        pool2 = Maxpool2D(conv2, f_shape = (2, 2))  # (?, 16, 28, 28)

        conv3 = Conv2D(pool2, self.kernel3, self.bias3, pad = 2)  # (?, 32, 28, 28)
        conv3 = tce.ReLU(conv3)

        pool3 = Maxpool2D(conv3, f_shape = (2, 2))  # (?, 32, 14, 14)

        conv4 = Conv2D(pool3, self.kernel4, self.bias4, pad = 1)  # (?, 32, 14, 14)
        conv4 = tce.ReLU(conv4)

        pool4 = Maxpool2D(conv4, f_shape = (2, 2))  # (?, 32, 7, 7)

        pool4 = pool4.reshape((-1, 1568))  # (?, 1568)

        fc1 = pool4 @ self.fc1w + self.fc1b  # (?, 64)
        fc1 = tce.ReLU(fc1)

        fc2 = fc1 @ self.fc2w + self.fc2b  # (?, 32)
        fc2 = tce.ReLU(fc2)

        fc3_1 = fc2 @ self.fc3w_1 + self.fc3b_1  # (?, 2)
        fc3_1 = tce.sigmoid(fc3_1)

        fc3_2 = fc2 @ self.fc3w_2 + self.fc3b_2  # (?, 4)
        self.fc3_2 = tce.sigmoid(fc3_2)

        return fc3_1  # (?, 2)


class Face_Loss(LOSS):

    def __init__(self):
        super().__init__(name = 'face model loss function')  # pass your loss's name

    def define_loss(self, predicted: 'Tensor', actual: 'Tensor', model: 'Module' = None) -> None:
        """
        describe loss function for model here.

        The Loss is consistent of two parts: label's error and board's error.

        label's error is difference between the predicted value of the image classification
        result and the actual value.
        In addition, if image contacts faces, the board's error would describe difference between
        predicted position of face and the actual position. Else, the board's error would be assign
        with 0.

        """
        alpha, beta = 1, 4  # weights of two error parts

        acc_label = actual[:, 0:2]  # （32， 2）
        acc_board = actual[:, 2:]  # （32， 4）
        acc_mark = actual[:, [0]]  # (32, 1)

        err_label = tce.sum((predicted - acc_label) ** 2, axis = 1, keepdims = True)
        err_label = tce.mean(err_label)  # mean square error describe the label's error

        err_board = tce.sum((model['fc3_2'] - acc_board) ** 2, axis = 1, keepdims = True)  # (32, 1)
        err_board = acc_mark * err_board  # (32, 1)
        err_board = tce.mean(err_board)

        self.loss = err_board * beta + err_label * alpha  # sum the error and assign `self.loss`.

        model.fc3_2 = None


class Face_Accuracy_Board(AccuracyBoard):
    def __init__(self):
        super().__init__('face model accuracy board')

    def define_accuracy(self, predict: Tensor, accuracy: Tensor, model: Module):

        predict.non_gradient()  # ensure predict is a non-gradient tensor
        p = predict.sum(axis = 1)

        accuracy.non_gradient()  # ensure accuracy is a non-gradient tensor
        a = accuracy[:, 0:2].sum(axis = 1)

        result = tce.abs((p - a)) < 1
        # if |predicted - actual| < constant, mark it classify correctly.
        self.accuracy = (1 * result).mean()  # compute mean accuracy and assign `self.accuracy`


def show_board(img, board):
    h, w, c = img.shape

    dx = w * board[0]
    dy = h * board[1]

    dw = w * board[2]
    dh = h * board[3]

    point_1 = (int(dx - dw / 2), int(dy + dh / 2))
    point_2 = (int(dx + dw / 2), int(dy - dh / 2))

    cv2.rectangle(img, point_1, point_2, (255, 0, 0), 2)
    # cv2.rectangle(img, (100, 100), (500, 1000), (0, 255, 0), 4)

    cv2.imshow('p', cv2.resize(img, (600, 600)))
    cv2.waitKey(0)

need_train = False
need_test = True
test_file = '../face/f9.png'


model_path = '../face/f_model1.pkl'

if os.path.exists(model_path):
    f_model = tce.loadmodel(model_path)  # load your saved model
    if need_train:
        f_loss = Face_Loss()  # instantiate your loss module
        f_acc= Face_Accuracy_Board()  # instantiate your acc module

        f_model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,  # choose adaptive moment estimation optimizer
                        loss = f_loss,  # replace with your loss
                        learning_rate = 0.005)

        f_model.fit(x_data, y_data,
                    batch_size = 32,
                    epochs = 10,
                    validation_split = 0.2,
                    show_loss = True,  # show loss of test set per epoch
                    show_acc_tr = True,  # show accuracy of training set and test set per epoch
                    epochs_mean_loss = True,  # show mean loss of batch
                    show_batch_loss = True,  # show loss of train per batch
                    show_batch_acc = True,  # show accuracy of train per batch
                    accuracy_show = f_acc)  # replace it with your acc_show

else:
    f_model = Face_Model()  # instantiate your model
    if need_train:
        f_loss = Face_Loss()  # instantiate your loss module
        f_acc = Face_Accuracy_Board()  # instantiate your acc module

        f_model.compile(optimizer = tce.nn.ADAM_OPTIMIZER,  # choose adaptive moment estimation optimizer
                        loss = f_loss,  # replace with your loss
                        learning_rate = 0.005)

        f_model.fit(x_data, y_data,
                    batch_size = 32,
                    epochs = 10,
                    validation_split = 0.2,
                    show_loss = True,  # show loss of test set per epoch
                    show_acc_tr = True,  # show accuracy of training set and test set per epoch
                    epochs_mean_loss = True,  # show mean loss of batch
                    show_batch_loss = True,  # show loss of train per batch
                    show_batch_acc = True,  # show accuracy of train per batch
                    accuracy_show = f_acc)  # replace it with your acc_show

if need_train:
    f_model.summary()
    tce.savemodel(f_model, model_path)


if need_test:
    img = cv2.imread(test_file)
    img_t = cv2.resize(img, (128, 128))
    img_t = Tensor(img_t).reshape((1, 128, 128, 3))
    img_t = img_t.transpose(0, 3, 1, 2)
    img_t /= 255

    res = f_model.predict(img_t)
    board = f_model['fc3_2'].reshape((4,))

    show_board(img, board.data)

    print('Probably: %4.2f%%' % (res[0, 1].data * 100))
