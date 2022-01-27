from .func import Conv2D
# noinspection PyDeprecation
from .func import convolution
from .func import Maxpool2D
from .func import dropout_layer
from .func import Meanpool2D
from .classes import Dataloader, Dataset

from .optim import SGD_OPTIMIZER, SGDM_OPTIMIZER, ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, RMSPROP_OPTIMIZER
from .optim import Optimizer
from .loss import LOSS
from .loss import CATEGORYLOSS, MSELOSS
from .layer import Layer_Manager, \
    ConnectionLayer, \
    Dense, \
    Convolution_layer2D, \
    Flatten, \
    Batch_normalize_layer, \
    MaxPooling_layer2D, \
    AveragePool_layer2D
from .layer import Activation
