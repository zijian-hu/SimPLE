from .cifar10_datamodule import CIFAR10DataModule
from .domainnet import DomainNet
from .utils import per_class_random_split

# for type hint
from typing import Optional, List
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class DomainNetDataModule(CIFAR10DataModule):
    num_classes: int = 345

    total_train_size: int = 120_906
    total_test_size: int = 52_041

    DATASET = DomainNet

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
        super(DomainNetDataModule, self).__init__(
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

        self.dims = (3, 84, 84)

        # dataset stats
        # DomainNet-Real mean, std values in CHW
        self.dataset_mean = [0.71566543, 0.74702915, 0.73697001]
        self.dataset_std = [0.32940442, 0.30978642, 0.30869427]

    def prepare_data(self, *args, **kwargs):
        self.DATASET(root=self.data_dir, train=True, split="real", download=True)
        self.DATASET(root=self.data_dir, train=False, split="real", download=True)

    def setup(self, stage: Optional[str] = None):
        full_train_set = self.DATASET(root=self.data_dir, train=True, split="real")
        full_test_set = self.DATASET(root=self.data_dir, train=False, split="real", transform=self.test_transforms)

        self.setup_helper(full_train_set=full_train_set, full_test_set=full_test_set, stage=stage)

    def split_dataset(self, dataset: Dataset, **kwargs) -> List[Dataset]:
        split_kwargs = dict(lengths=[self.validation_size, self.labeled_train_size],
                            num_classes=self.num_classes,
                            uneven_split=True)

        # update split arguments
        split_kwargs.update(kwargs)

        return per_class_random_split(dataset, **split_kwargs)
