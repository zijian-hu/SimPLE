from PIL import Image
import h5py
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive
from pathlib import Path


class MiniImageNet(VisionDataset):
    base_folder = 'mini-imagenet'

    # file info
    gdrive_id = '1EKmnUcpipszzBHBRcXxmejuO4pceD4ht'
    gdrive_md5 = '3bda5120eb7353dd88e06de46e680146'
    filename = 'miniimagenet.hdf5'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.root = Path(root)
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        img_key = 'train_img' if self.train else 'test_img'
        target_key = 'train_target' if self.train else 'test_target'
        with h5py.File(self.root / self.base_folder / self.filename, "r", swmr=True) as h5_f:
            self.data = h5_f[img_key][...]
            self.target = h5_f[target_key][...]

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
        download_file_from_google_drive(file_id=self.gdrive_id, root=self.root / self.base_folder,
                                        filename=self.filename, md5=self.gdrive_md5)

    def _check_integrity(self):
        return check_integrity(fpath=self.root / self.base_folder / self.filename, md5=self.gdrive_md5)
