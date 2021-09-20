from torchvision import transforms
from torch import distributed
from PIL import Image

from .utils import repeater, get_batch

from .miniimagenet import MiniImageNet
from .domainnet_real import DomainNetReal
from .datasets import LabeledDataset

from .datamodule import DataModule
from .ssl_datamodule import SSLDataModule

from .cifar10_datamodule import CIFAR10DataModule
from .cifar100_datamodule import CIFAR100DataModule
from .svhn_datamodule import SVHNDataModule
from .miniimagenet_datamodule import MiniImageNetDataModule
from .domainnet_real_datamodule import DomainNetRealDataModule

from ..transforms import CenterResizedCrop

# modules
from . import utils

# for type hint
from typing import Dict, Callable
from argparse import Namespace


def _get_world_size() -> int:
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def _get_dataset_transforms(args: Namespace) -> Dict[str, Callable]:
    if args.data_dims is None and args.dataset == "domainnet-real":
        dims = (3, 224, 224)
    else:
        dims = args.data_dims

    if dims is not None:
        dims = tuple(dims)

        return dict(
            dims=dims,
            train_transform=transforms.Compose([
                transforms.RandomResizedCrop(
                    size=dims[-2:],
                    scale=(0.08, 1.0),
                    ratio=(3. / 4, 4. / 3.),
                    interpolation=Image.BICUBIC),
                transforms.ToTensor()]),
            val_transform=transforms.Compose([
                CenterResizedCrop(dims[-2:], interpolation=Image.BICUBIC),
                transforms.ToTensor()]),
            test_transform=transforms.Compose([
                CenterResizedCrop(dims[-2:], interpolation=Image.BICUBIC),
                transforms.ToTensor()]),
        )

    else:
        return dict(
            train_transform=transforms.Compose([transforms.ToTensor()]),
            val_transform=transforms.Compose([transforms.ToTensor()]),
            test_transform=transforms.Compose([transforms.ToTensor()]),
        )


def get_dataset(args: Namespace) -> SSLDataModule:
    world_size = _get_world_size()

    kwargs = dict(
        data_dir=args.data_dir,
        labeled_train_size=args.labeled_train_size,
        validation_size=args.validation_size,
        train_batch_size=args.train_batch_size,
        unlabeled_batch_size=args.unlabeled_train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        train_min_size=world_size * args.train_batch_size,
        unlabeled_train_min_size=world_size * args.unlabeled_train_batch_size,
        test_min_size=world_size * args.test_batch_size,
        **_get_dataset_transforms(args),
    )

    if args.dataset == "cifar10":
        return CIFAR10DataModule(**kwargs)

    elif args.dataset == "cifar100":
        return CIFAR100DataModule(**kwargs)

    elif args.dataset == "svhn":
        return SVHNDataModule(**kwargs)

    elif args.dataset == "miniimagenet":
        return MiniImageNetDataModule(**kwargs)

    elif args.dataset == "domainnet-real":
        return DomainNetRealDataModule(**kwargs)

    else:
        raise NotImplementedError(f"\"{args.dataset}\" is not a supported dataset")


__all__ = [
    # modules
    "utils",
    "types",

    # classes
    "MiniImageNet",
    "DomainNetReal",
    "LabeledDataset",

    "DataModule",
    "SSLDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "DomainNetRealDataModule",
    "MiniImageNetDataModule",
    "SVHNDataModule",

    # functions
    "repeater",
    "get_batch",
    "get_dataset",
]
