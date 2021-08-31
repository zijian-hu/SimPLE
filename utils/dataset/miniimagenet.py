import h5py
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive

from pathlib import Path

# for type hint
from typing import Optional, Callable


class MiniImageNet(VisionDataset):
    base_folder = 'mini-imagenet'
    gdrive_id = '1EKmnUcpipszzBHBRcXxmejuO4pceD4ht'
    file_md5 = '3bda5120eb7353dd88e06de46e680146'
    filename = 'mini-imagenet.hdf5'

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        img_key = 'train_img' if self.train else 'test_img'
        target_key = 'train_target' if self.train else 'test_target'
        with h5py.File(self.data_root / self.filename, "r", swmr=True) as h5_f:
            self.data = h5_f[img_key][...]
            self.target = h5_f[target_key][...]

    @property
    def data_root(self) -> Path:
        return Path(self.root) / self.base_folder

    @property
    def download_root(self) -> Path:
        return self.data_root

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img, target = Image.fromarray(self.data[idx]), self.target[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_file_from_google_drive(file_id=self.gdrive_id,
                                        root=str(self.download_root),
                                        filename=self.filename,
                                        md5=self.file_md5)

    def _check_integrity(self):
        return check_integrity(fpath=str(self.download_root / self.filename), md5=self.file_md5)
