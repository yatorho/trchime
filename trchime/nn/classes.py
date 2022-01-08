from ..core.tensor import Tensor

class Dataloader:
    def __init__(self,
                 dataset: 'Dataset' = None,
                 batch_size: int = 32,
                 shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


class Dataset:
    pass
