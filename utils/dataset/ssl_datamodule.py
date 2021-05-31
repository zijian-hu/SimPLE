from abc import ABC

from .datamodule import DataModule
from .utils import get_batch

# for type hint
from typing import Optional, Tuple, List, Union
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from .utils import BatchGeneratorType


class SSLDataModule(DataModule, ABC):
    num_classes: int = 1

    total_train_size: int = 1
    total_test_size: int = 1

    DATASET = type(Dataset)

    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 unlabeled_train_batch_size: Optional[int] = None,
                 test_batch_size: Optional[int] = None,
                 train_transforms: Optional[Compose] = None,
                 val_transforms: Optional[Compose] = None,
                 test_transforms: Optional[Compose] = None):
        super(SSLDataModule, self).__init__(train_transforms=train_transforms,
                                            val_transforms=val_transforms,
                                            test_transforms=test_transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unlabeled_batch_size = unlabeled_train_batch_size
        self.test_batch_size = test_batch_size

        self.dims: Optional[Tuple[int, ...]] = None

        self.labeled_train_set: Optional[Dataset] = None
        self.unlabeled_train_set: Optional[Dataset] = None
        self.validation_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

        # dataset stats
        self.dataset_mean: Optional[List[float]] = None
        self.dataset_std: Optional[List[float]] = None

    @property
    def unlabeled_batch_size(self) -> int:
        return self._unlabeled_batch_size if self._unlabeled_batch_size is not None else self.batch_size

    @unlabeled_batch_size.setter
    def unlabeled_batch_size(self, batch_size: int):
        self._unlabeled_batch_size = batch_size

    @property
    def test_batch_size(self) -> int:
        return self._test_batch_size if self._test_batch_size is not None else self.batch_size

    @test_batch_size.setter
    def test_batch_size(self, batch_size: int):
        self._test_batch_size = batch_size

    def train_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [DataLoader(self.labeled_train_set,
                           shuffle=True,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           drop_last=True,
                           **kwargs),
                DataLoader(self.unlabeled_train_set,
                           shuffle=True,
                           batch_size=self.unlabeled_batch_size,
                           num_workers=self.num_workers,
                           drop_last=True,
                           **kwargs)]

    def val_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_set,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **kwargs)

    def test_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **kwargs)

    def get_train_batch(self, train_loaders: List[DataLoader], **kwargs) -> BatchGeneratorType:
        return get_batch(train_loaders, **kwargs)
