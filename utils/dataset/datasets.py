from torch.utils.data import Dataset, Subset

# for type hint
from torchvision.datasets import VisionDataset
from typing import Union, Optional, Callable


class LabeledDataset(VisionDataset):
    def __init__(self,
                 dataset: Union[Dataset, Subset],
                 root: str,
                 min_size: int = 0,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transforms, transform, target_transform)

        self.dataset = dataset

        self.min_size = min_size

    @property
    def min_size(self) -> int:
        return self._min_size if len(self.dataset) > 0 else 0

    @min_size.setter
    def min_size(self, min_size: int) -> None:
        if min_size < 0:
            raise ValueError(f"only non-negative min_size is allowed")

        self._min_size = min_size

    def __getitem__(self, index: int):
        img, target = self.dataset[index % len(self.dataset)]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return max(len(self.dataset), self.min_size)
