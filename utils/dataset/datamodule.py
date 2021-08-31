from abc import ABC, abstractmethod

# for type hint
from typing import Optional, Tuple, Callable, Union, List, Generator, Any
from torch.utils.data import DataLoader


class DataModule(ABC):
    def __init__(self,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 dims: Optional[Tuple[int, ...]] = None):
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.dims = dims

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
