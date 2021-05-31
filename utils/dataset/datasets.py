import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

# for type hint
from torchvision.datasets import VisionDataset
from typing import Optional, Generator, Tuple, Union, List, Sequence, Any


class LabeledDataset(VisionDataset):
    def __init__(self, dataset: Union[Dataset, Subset], root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

        self.dataset = dataset

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)


class UnlabeledDataset(LabeledDataset):
    def __init__(self, dataset: Union[Dataset, Subset], root, transforms=None, transform=None, target_transform=None):
        super().__init__(dataset, root, transforms, transform, target_transform)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, -1


class ConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Union[Dataset, Subset]], max_length: Optional[int] = None,
                 use_random_map: bool = False):
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/1089
        assert max_length is None or max_length >= 0
        self._datasets = datasets
        self._use_random_map = use_random_map

        self.is_max_length_interpolated = max_length is None
        self.max_length = max_length if not self.is_max_length_interpolated else 0
        self.lengths: Optional[np.ndarray] = None
        self.index_map: Optional[np.ndarray] = None

        self._update_stats()
        self._update_index_map()

    @property
    def datasets(self) -> List[Union[Dataset, Subset]]:
        return list(self._datasets)

    @datasets.setter
    def datasets(self, _datasets):
        self._datasets = _datasets

        self._update_stats()
        self._update_index_map()

    @property
    def use_random_map(self):
        return self._use_random_map

    @use_random_map.setter
    def use_random_map(self, _use_random_map):
        self._use_random_map = _use_random_map

        # update index map with random order
        self._update_index_map()

    def _update_stats(self):
        self.lengths = np.asarray([len(d) for d in self.datasets])

        if self.is_max_length_interpolated:
            self.max_length = np.max(self.lengths)

        self.index_map = np.zeros((len(self.lengths), self.max_length), dtype=int)

    def _update_index_map(self):
        self.index_map[:, range(self.max_length)] = range(self.max_length)
        self.index_map %= self.lengths.reshape(-1, 1)

        if self._use_random_map:
            reset_idx = np.where(self.lengths >= self.max_length)
            reset_arr = self.index_map[reset_idx]

            shuffle_idx = np.arange(self.max_length)
            np.random.shuffle(shuffle_idx)

            self.index_map = self.index_map[:, shuffle_idx]
            self.index_map[reset_idx] = reset_arr

    def __getitem__(self, i):
        return tuple(d[m[i]] for d, m in zip(self.datasets, self.index_map))

    def __len__(self):
        # will be called every epoch
        # self._update_index_map()
        return self.max_length
