import numpy as np
import os

from ..core.tensor import Tensor


class data_set:
    def __init__(self, path):
        self.path = path

    def load(self, whether_tensor: bool):
        pass


class mnist_set(data_set):
    def __init__(self, path):
        super().__init__(path)

    def load(self, whether_tensor: bool = False):

        if os.path.exists(self.path):
            file = np.load(self.path)

            dataset = [None, None, None, None]

            for i, k in enumerate(file.keys()):
                dataset[i] = file[k]

                if whether_tensor:
                    dataset[i] = Tensor(dataset[i])

            return tuple(dataset)

        raise AttributeError('invalided path')

class cifar_set(data_set):
    def __init__(self, path):
        super().__init__(path)

    def load(self, whether_tensor: bool = False):

        if os.path.exists(self.path):
            file = np.load(self.path)

            dataset = [None, None, None, None]

            for i, k in enumerate(file.keys()):
                dataset[i] = file[k]

                if whether_tensor:
                    dataset[i] = Tensor(dataset[i])

            return tuple(dataset)

        raise AttributeError('invalided path')


class faces_set(data_set):
    def __init__(self, path):
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
        super().__init__(path)

    def load(self, whether_tensor: bool = False):
        if os.path.exists(self.path):
            file = np.load(self.path)

            dataset = [None, None]

            for i, k in enumerate(file.keys()):
                dataset[i] = file[k]

                if whether_tensor:
                    dataset[i] = Tensor(dataset[i])

            return tuple(dataset)

        raise AttributeError('invalided path')
