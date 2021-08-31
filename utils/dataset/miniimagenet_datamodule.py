from .miniimagenet import MiniImageNet
from .cifar10_datamodule import CIFAR10DataModule

# for type hint
from typing import Optional, Tuple


class MiniImageNetDataModule(CIFAR10DataModule):
    num_classes: int = 100

    total_train_size: int = 50_000
    total_test_size: int = 10_000

    DATASET = MiniImageNet

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 dims: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        if dims is None:
            dims = (3, 84, 84)

        super(MiniImageNetDataModule, self).__init__(
            data_dir=data_dir,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            dims=dims,
            **kwargs)

        # dataset stats
        # Mini-ImageNet mean, std values in CHW
        self.dataset_mean = [0.40233998, 0.47269102, 0.44823737]
        self.dataset_std = [0.2884859, 0.28327602, 0.27511246]
