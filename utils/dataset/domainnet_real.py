from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_url, extract_archive

from pathlib import Path
from typing import NamedTuple
import re
import os
import shutil

from .utils import get_directory_size

# for type hint
from typing import Optional

FileMeta = NamedTuple("FileMeta", [("filename", str), ("url", str), ("md5", Optional[str])])


class DomainNetReal(ImageFolder):
    base_folder = 'domainnet-real'

    data_file_meta = FileMeta(
        filename="domainnet-real.zip",
        url="http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        md5="dcc47055e8935767784b7162e7c7cca6")

    train_label_file_meta = FileMeta(
        filename="domainnet-real_train.txt",
        url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
        md5="8ebf02c2075fadd564705f0dc7cd6291")

    test_label_file_meta = FileMeta(
        filename="domainnet-real_test.txt",
        url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
        md5="6098816791c3ebed543c71ffa11b9054")

    label_file_pattern = re.compile(r"(.*) (\d+)")

    # extracted file size in Bytes
    DATA_FOLDER_SIZE = 6_234_186_058
    TRAIN_FOLDER_SIZE = 4_301_431_405
    TEST_FOLDER_SIZE = 1_860_180_803

    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = False,
                 **kwargs):
        self.root = root
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # parse dataset
        self.parse_archives()

        super(DomainNetReal, self).__init__(root=str(self.split_folder), **kwargs)
        self.root = root

    @property
    def data_root(self) -> Path:
        return Path(self.root) / self.base_folder

    @property
    def download_root(self) -> Path:
        return Path(self.root)

    @property
    def data_folder(self) -> Path:
        return self.data_root / "real"

    @property
    def split_folder(self) -> Path:
        return self.data_root / ("train" if self.train else "test")

    @property
    def SPLIT_FOLDER_SIZE(self) -> int:
        return self.TRAIN_FOLDER_SIZE if self.train else self.TEST_FOLDER_SIZE

    @property
    def label_file_meta(self) -> FileMeta:
        return self.train_label_file_meta if self.train else self.test_label_file_meta

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # remove old files
        shutil.rmtree(self.split_folder, ignore_errors=True)
        shutil.rmtree(self.data_folder, ignore_errors=True)

        for file_meta in (self.data_file_meta, self.label_file_meta):
            download_url(url=file_meta.url,
                         root=str(self.download_root),
                         filename=file_meta.filename,
                         md5=file_meta.md5)

    def _check_integrity(self) -> bool:
        for file_meta in (self.data_file_meta, self.label_file_meta):
            if not check_integrity(fpath=str(self.download_root / file_meta.filename), md5=file_meta.md5):
                return False

        return True

    def parse_archives(self) -> None:
        if not self.split_folder.is_dir() or get_directory_size(self.split_folder) < self.SPLIT_FOLDER_SIZE:
            # if split_folder do not exist or not large enough
            self.parse_data_archive()

            # remove old files
            shutil.rmtree(self.split_folder, ignore_errors=True)

            with open(self.download_root / self.label_file_meta.filename, "r") as f:
                file_content = [line.strip() for line in f.readlines()]

            for line in file_content:
                search_result = self.label_file_pattern.search(line)
                assert search_result is not None, f"{self.label_file_meta.filename} contains invalid line \"{line}\""

                image_path = Path(search_result.group(1))
                image_relative_path = Path(*image_path.parts[1:])

                source_path = self.data_root / image_path
                target_path = self.split_folder / image_relative_path

                if not target_path.is_file():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(src=source_path.absolute(), dst=target_path.absolute())

    def parse_data_archive(self) -> None:
        if not self.data_folder.is_dir() or get_directory_size(self.data_folder) < self.DATA_FOLDER_SIZE:
            # if data_folder do not exist or not large enough
            # remove old files
            shutil.rmtree(self.data_folder, ignore_errors=True)

            print(f"extracting {self.data_file_meta.filename}...")
            extract_archive(from_path=str(self.download_root / self.data_file_meta.filename),
                            to_path=str(self.data_root))
