import torch
from torch import nn
from kornia import augmentation as K
from kornia import filters as F
from torchvision import transforms

from .augmenter import RandomAugmentation
from .randaugment import RandAugmentNS

# for type hint
from typing import List, Tuple, Union, Callable
from torch import Tensor
from torch.nn import Module
from PIL.Image import Image as PILImage

DatasetStatType = List[float]
ImageSizeType = Tuple[int, int]
PaddingInputType = Union[float, Tuple[float, float], Tuple[float, float, float, float]]
ImageType = Union[Tensor, PILImage]


def get_augmenter(augmenter_type: str,
                  image_size: ImageSizeType,
                  dataset_mean: DatasetStatType,
                  dataset_std: DatasetStatType,
                  padding: PaddingInputType = 1. / 8.,
                  pad_if_needed: bool = False,
                  subset_size: int = 2) -> Union[Module, Callable]:
    """
    
    Args:
        augmenter_type: augmenter type
        image_size: (height, width) image size
        dataset_mean: dataset mean value in CHW
        dataset_std: dataset standard deviation in CHW
        padding: percent of image size to pad on each border of the image. If a sequence of length 4 is provided,
            it is used to pad left, top, right, bottom borders respectively. If a sequence of length 2 is provided, it is
            used to pad left/right, top/bottom borders, respectively.
        pad_if_needed: bool flag for RandomCrop "pad_if_needed" option
        subset_size: number of augmentations used in subset

    Returns: nn.Module for Kornia augmentation or Callable for torchvision transform

    """
    if not isinstance(padding, tuple):
        assert isinstance(padding, float)
        padding = (padding, padding, padding, padding)

    assert len(padding) == 2 or len(padding) == 4
    if len(padding) == 2:
        # padding of length 2 is used to pad left/right, top/bottom borders, respectively
        # padding of length 4 is used to pad left, top, right, bottom borders respectively
        padding = (padding[0], padding[1], padding[0], padding[1])

    # image_size is of shape (h,w); padding values is [left, top, right, bottom] borders
    padding = (
        int(image_size[1] * padding[0]),
        int(image_size[0] * padding[1]),
        int(image_size[1] * padding[2]),
        int(image_size[0] * padding[3])
    )

    augmenter_type = augmenter_type.strip().lower()

    if augmenter_type == "simple":
        return nn.Sequential(
            K.RandomCrop(size=image_size, padding=padding, pad_if_needed=pad_if_needed,
                         padding_mode='reflect'),
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(mean=torch.tensor(dataset_mean, dtype=torch.float32),
                        std=torch.tensor(dataset_std, dtype=torch.float32)),
        )

    elif augmenter_type == "fixed":
        return nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.2),
            K.RandomResizedCrop(size=image_size, scale=(0.8, 1.0), ratio=(1., 1.)),
            RandomAugmentation(
                p=0.5,
                augmentation=F.GaussianBlur2d(
                    kernel_size=(3, 3),
                    sigma=(1.5, 1.5),
                    border_type='constant'
                )
            ),
            K.ColorJitter(contrast=(0.75, 1.5)),
            # additive Gaussian noise
            K.RandomErasing(p=0.1),
            # Multiply
            K.RandomAffine(
                degrees=(-25., 25.),
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=(-8., 8.)
            ),
            K.Normalize(mean=torch.tensor(dataset_mean, dtype=torch.float32),
                        std=torch.tensor(dataset_std, dtype=torch.float32)),
        )

    elif augmenter_type in ["validation", "test"]:
        return nn.Sequential(
            K.Normalize(mean=torch.tensor(dataset_mean, dtype=torch.float32),
                        std=torch.tensor(dataset_std, dtype=torch.float32)),
        )

    elif augmenter_type == "randaugment":
        return nn.Sequential(
            K.RandomCrop(size=image_size, padding=padding, pad_if_needed=pad_if_needed,
                         padding_mode='reflect'),
            K.RandomHorizontalFlip(p=0.5),
            RandAugmentNS(n=subset_size, m=10),
            K.Normalize(mean=torch.tensor(dataset_mean, dtype=torch.float32),
                        std=torch.tensor(dataset_std, dtype=torch.float32)),
        )

    else:
        raise NotImplementedError(f"\"{augmenter_type}\" is not a supported augmenter type")


__all__ = [
    # modules
    # classes
    # functions
    "get_augmenter",
]
