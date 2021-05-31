from torchvision.datasets import CIFAR100

from .cifar10_datamodule import CIFAR10DataModule

# for type hint
from typing import Optional
from torchvision.transforms import Compose


class CIFAR100DataModule(CIFAR10DataModule):
    num_classes: int = 100

    total_train_size: int = 50_000
    total_test_size: int = 10_000

    DATASET = CIFAR100

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 unlabeled_train_batch_size: Optional[int] = None,
                 test_batch_size: Optional[int] = None,
                 train_transforms: Optional[Compose] = None,
                 val_transforms: Optional[Compose] = None,
                 test_transforms: Optional[Compose] = None):
        super(CIFAR100DataModule, self).__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            unlabeled_train_batch_size=unlabeled_train_batch_size,
            test_batch_size=test_batch_size,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms)

        self.dims = (3, 32, 32)

        # dataset stats
        # CIFAR-100 mean, std values in CHW
        self.dataset_mean = [0.44091784, 0.50707516, 0.48654887]
        self.dataset_std = [0.27615047, 0.26733429, 0.25643846]
