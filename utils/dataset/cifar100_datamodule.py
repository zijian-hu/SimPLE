from torchvision.datasets import CIFAR100

from .cifar10_datamodule import CIFAR10DataModule

# for type hint
from typing import Optional


class CIFAR100DataModule(CIFAR10DataModule):
    num_classes: int = 100

    total_train_size: int = 50_000
    total_test_size: int = 10_000

    DATASET = CIFAR100

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 **kwargs):
        super(CIFAR100DataModule, self).__init__(
            data_dir=data_dir,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            **kwargs)

        # dataset stats
        # CIFAR-100 mean, std values in CHW
        self.dataset_mean = [0.44091784, 0.50707516, 0.48654887]
        self.dataset_std = [0.27615047, 0.26733429, 0.25643846]
