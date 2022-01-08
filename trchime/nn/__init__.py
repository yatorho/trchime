from .func import convolution_layer2D
# noinspection PyDeprecation
from .func import convolution
from .func import maxpooling_layer2D
from .func import dropout_layer
from .func import meanpooling_layer2D
from .classes import Dataloader, Dataset

from .optim import SGD_OPTIMIZER, SGDM_OPTIMIZER, ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, RMSPROP_OPTIMIZER
from .loss import CATEGORYLOSS, MSELOSS
