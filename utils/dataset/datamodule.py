from abc import ABC, abstractmethod

# for type hint
from typing import List, Union, Optional, Any, Generator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class DataModule(ABC):
    def __init__(self,
                 train_transforms: Optional[Compose] = None,
                 val_transforms: Optional[Compose] = None,
                 test_transforms: Optional[Compose] = None,
                 dims=None):
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    @abstractmethod
    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    @abstractmethod
    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    @abstractmethod
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def get_train_batch(self, *args, **kwargs) -> Generator[Any, Any, Any]:
        pass
