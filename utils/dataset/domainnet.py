from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, extract_archive, check_integrity

from typing import NamedTuple
from pathlib import Path
import os
import re

# for type hint
from typing import Optional, Union, Collection, Set, Dict
from collections.abc import Iterable

FileInfo = NamedTuple("FileInfo", [("filename", str), ("url", str), ("md5", Optional[str])])
LabelInfo = NamedTuple("LabelInfo", [("source_path", Path), ("target_path", Path), ("label", str)])

DatasetSplitType = Optional[Union[str, Collection[str]]]


class DomainNet(ImageFolder):
    base_folder = 'domain-net'

    data_file_info = {
        "clipart": FileInfo(
            filename="clipart.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            md5="cd0d8f2d77a4e181449b78ed62bccf1e"),
        "infograph": FileInfo(
            filename="infograph.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
            md5="720380b86f9e6ab4805bb38b6bd135f8"),
        "painting": FileInfo(
            filename="painting.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
            md5="1ae32cdb4f98fe7ab5eb0a351768abfd"),
        "quickdraw": FileInfo(
            filename="quickdraw.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
            md5="bdc1b6f09f277da1a263389efe0c7a66"),
        "real": FileInfo(
            filename="real.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
            md5="dcc47055e8935767784b7162e7c7cca6"),
        "sketch": FileInfo(
            filename="sketch.zip",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
            md5="658d8009644040ff7ce30bb2e820850f"),
    }

    train_label_file_info = {
        "clipart": FileInfo(
            filename="clipart_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
            md5=None),
        "infograph": FileInfo(
            filename="infograph_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
            md5=None),
        "painting": FileInfo(
            filename="painting_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
            md5=None),
        "quickdraw": FileInfo(
            filename="quickdraw_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
            md5=None),
        "real": FileInfo(
            filename="real_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
            md5=None),
        "sketch": FileInfo(
            filename="sketch_train.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
            md5=None),
    }

    test_label_file_info = {
        "clipart": FileInfo(
            filename="clipart_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt",
            md5=None),
        "infograph": FileInfo(
            filename="infograph_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt",
            md5=None),
        "painting": FileInfo(
            filename="painting_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt",
            md5=None),
        "quickdraw": FileInfo(
            filename="quickdraw_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt",
            md5=None),
        "real": FileInfo(
            filename="real_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
            md5=None),
        "sketch": FileInfo(
            filename="sketch_test.txt",
            url="http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt",
            md5=None),
    }

    def __init__(self, root: str, train: bool = True, split: DatasetSplitType = None, transform=None,
                 target_transform=None, loader=default_loader, is_valid_file=None, download: bool = False):
        self.train = train

        self.root_dir_path = Path(root)
        self.base_dir_path = self.root_dir_path / self.base_folder

        if self.train:
            self.label_file_info = self.train_label_file_info
            self.output_dir_path = self.base_dir_path / "train"
        else:
            self.label_file_info = self.test_label_file_info
            self.output_dir_path = self.base_dir_path / "test"

        self.split = split
        self._subsets = None
        self._label_info_dict: Dict[str, Set[LabelInfo]] = dict()

        if download:
            self.download()

        root = str(self.output_dir_path)

        super().__init__(root, transform, target_transform, loader, is_valid_file)

    @staticmethod
    def split_to_subsets(split: DatasetSplitType) -> Set[str]:
        full_sets = DomainNet.data_file_info.keys()

        if split is None:
            return set(full_sets)

        if isinstance(split, str):
            return {split}

        if isinstance(split, Iterable):
            subsets = set(split)
            # check if all subsets are supported
            unexpected_subsets = subsets - full_sets
            assert len(unexpected_subsets) == 0, f"{unexpected_subsets} are unexpected subsets"

            return subsets

        # if type is not supported
        raise RuntimeError(f"unexpected type of split {type(split)}")

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # remove old links
        self._remove_link(is_remove_all=True)

        for subset in self.subsets:
            for info_dict in [self.data_file_info, self.label_file_info]:
                assert subset in info_dict
                file_info = info_dict[subset]

                download_url(url=file_info.url, root=str(self.root_dir_path), filename=file_info.filename,
                             md5=file_info.md5)

        self.base_dir_path.mkdir(parents=False, exist_ok=True)

        # unzip
        for subset in self.subsets:
            if not self._check_subset_integrity(subset, is_check_link=False):
                data_file_path = self.root_dir_path / self.data_file_info[subset].filename
                print(f"unzipping {data_file_path}...")
                extract_archive(str(data_file_path), self.base_dir_path)

        # create link
        for subset in self.subsets:
            if self._check_subset_integrity(subset, is_check_link=True):
                continue

            self._create_link(subset)

    def _check_integrity(self) -> bool:
        # remove unrelated files
        self._remove_link(is_remove_all=False)

        for subset in self.subsets:
            # check raw file integrity
            for info_dict in [self.data_file_info, self.label_file_info]:
                assert subset in info_dict
                file_info = info_dict[subset]
                filename = str(self.root_dir_path / file_info.filename)

                if not check_integrity(filename, file_info.md5):
                    return False

            # check unzipped file integrity
            if not self._check_subset_integrity(subset, is_check_link=True):
                return False

        return True

    def _check_subset_integrity(self, subset: str, is_check_link: bool = False) -> bool:
        subset_label_info = self.label_info_dict[subset]

        for label_info in subset_label_info:
            if not label_info.source_path.is_file():
                return False

            if is_check_link:
                if not label_info.target_path.is_file():
                    return False

        return True

    def _create_link(self, subset: str):
        print(f"creating links for {subset}...")

        # create link
        self.output_dir_path.mkdir(parents=False, exist_ok=True)

        subset_label_info = self.label_info_dict[subset]
        for label_info in subset_label_info:
            source_path = label_info.source_path
            target_path = label_info.target_path

            if target_path.is_file():
                continue

            target_path.parent.mkdir(parents=False, exist_ok=True)

            os.link(src=source_path.absolute(), dst=target_path.absolute())

    def _convert_file_path(self, file_path: Union[str, Path]) -> Path:
        return self.output_dir_path / "/".join(Path(file_path).parts[-2:])

    def _remove_link(self, is_remove_all: bool = False):
        if not self.output_dir_path.is_dir():
            return

        paths = set(self.output_dir_path.rglob("*"))
        files = {p for p in paths if p.is_file()}
        dirs = {p for p in paths if p.is_dir()}

        if is_remove_all:
            print("removing old links...")
            files_to_remove = files
            dirs_to_remove = dirs
        else:
            print("cleaning up links...")
            files_to_keep = {label_info.target_path for subset_label_info in self.label_info_dict.values()
                             for label_info in subset_label_info}
            dirs_to_keep = {file_path.parent for file_path in files_to_keep}
            files_to_remove = files - files_to_keep
            dirs_to_remove = dirs - dirs_to_keep

        for file_to_remove in files_to_remove:
            if file_to_remove.is_file():
                file_to_remove.unlink()

        # remove empty dirs
        empty_dirs = {p for p in dirs if p.is_dir() and len(os.listdir(p)) == 0}
        dirs_to_remove = dirs_to_remove.union(empty_dirs)

        for dir_to_remove in dirs_to_remove:
            if dir_to_remove.is_dir():
                dir_to_remove.rmdir()

    @property
    def subsets(self) -> Set[str]:
        if self._subsets is None:
            self._subsets = self.split_to_subsets(self.split)

        return self._subsets

    @property
    def label_info_dict(self) -> Dict[str, Set[LabelInfo]]:
        if len(self._label_info_dict) == 0:
            # download all label files
            for subset in self.subsets:
                assert subset in self.label_file_info
                file_info = self.label_file_info[subset]

                download_url(url=file_info.url, root=str(self.root_dir_path), filename=file_info.filename,
                             md5=file_info.md5)

                self._label_info_dict[subset] = set()

            # parse label file
            label_path_pattern = re.compile(r"(.*) (\d+)")

            for subset in self.subsets:
                file_info = self.label_file_info[subset]
                label_subset_info = self._label_info_dict[subset]

                filename = str(self.root_dir_path / file_info.filename)

                with open(filename, "r") as f:
                    content = f.readlines()

                content = [line.strip() for line in content]
                for line in content:
                    search_result = label_path_pattern.search(line)
                    assert search_result is not None, f"{filename} contains invalid line \"{line}\""

                    data_file_path = Path(self.base_dir_path) / search_result.group(1)
                    label = search_result.group(2)

                    label_info = LabelInfo(source_path=data_file_path,
                                           target_path=self._convert_file_path(data_file_path),
                                           label=label)

                    label_subset_info.add(label_info)

        return self._label_info_dict
