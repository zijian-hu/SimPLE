import torch
from torch import nn
from kornia import augmentation as K
from kornia import filters as F
import numpy as np

from random import random

# for type hint
from torch import Tensor
from typing import Tuple, Sequence, Optional, Union

DatasetStatType = Optional[Union[Tensor, float]]
ImageSizeType = Tuple[int, int]
PaddingInputType = Union[float, Tuple[float, float], Tuple[float, float, float, float]]


class BaseAugmenter(nn.Module):
    def __init__(self, image_size: ImageSizeType, dataset_mean: DatasetStatType, dataset_std: DatasetStatType,
                 padding: PaddingInputType = 0.125, pad_if_needed: bool = False):
        """

        :param image_size: (height, width) image size
        :param dataset_mean: dataset mean value for each channel in CHW
        :param dataset_std: dataset standard deviation for each channel in CHW
        :param padding: percent of image size to pad on each border of the image. If a sequence of length 4 is provided,
        it is used to pad left, top, right, bottom borders respectively. If a sequence of length 2 is provided, it is
        used to pad left/right, top/bottom borders, respectively.
        :param pad_if_needed: bool flag for RandomCrop "pad_if_needed" option
        """
        super().__init__()

        if dataset_mean is None:
            dataset_mean = torch.tensor([0.5] * 3, dtype=torch.float32)
        if dataset_std is None:
            dataset_std = torch.tensor([0.5] * 3, dtype=torch.float32)

        if not isinstance(padding, tuple):
            assert isinstance(padding, float)
            padding = (padding, padding, padding, padding)

        assert len(padding) == 2 or len(padding) == 4
        if len(padding) == 2:
            # padding of length 2 is used to pad left/right, top/bottom borders, respectively
            # padding of length 4 is used to pad left, top, right, bottom borders respectively
            padding = (padding[0], padding[1], padding[0], padding[1])

        self.image_size = image_size
        self.dataset_mean: Tensor = dataset_mean
        self.dataset_std: Tensor = dataset_std

        # image_size is of shape (h,w); padding values is [left, top, right, bottom] borders
        self.padding = (
            int(self.image_size[1] * padding[0]),
            int(self.image_size[0] * padding[1]),
            int(self.image_size[1] * padding[2]),
            int(self.image_size[0] * padding[3])
        )
        self.pad_if_needed = pad_if_needed

    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError


class RandomAugmentation(nn.Module):
    def __init__(self, augmentation: nn.Module, p: float = 0.5, same_on_batch: bool = False):
        super().__init__()

        self.prob = p
        self.augmentation = augmentation
        self.same_on_batch = same_on_batch

    def forward(self, images: Tensor) -> Tensor:
        is_batch = len(images) < 4

        if not is_batch or self.same_on_batch:
            if random() <= self.prob:
                out = self.augmentation(images)
            else:
                out = images
        else:
            out = self.augmentation(images)
            batch_size = len(images)

            # get the indices of data which shouldn't apply augmentation
            indices = torch.where(torch.rand(batch_size) > self.prob)
            out[indices] = images[indices]

        return out


class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int], min_sigma=0.1, max_sigma=2.0, p=0.5) -> None:
        super().__init__()
        self.kernel_size = tuple(s + 1 if s % 2 == 0 else s for s in kernel_size)  # kernel size must be odd
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p = p

    def forward(self, img):
        if self.p > random():
            sigma = (self.max_sigma - self.min_sigma) * random() + self.min_sigma
            return F.gaussian_blur2d(img, kernel_size=self.kernel_size, sigma=(sigma, sigma))
        else:
            return img


class RandomChoice(nn.Module):
    def __init__(self, augmentations: Sequence[nn.Module], size: int = 2, p: Optional[Sequence[float]] = None):
        super().__init__()

        assert size <= len(augmentations), f"size = {size} should be <= # aug. = {len(augmentations)}"

        self.augmentations = augmentations
        self.size = size
        self.p = p

    def forward(self, inputs: Tensor) -> Tensor:
        indices = np.random.choice(range(len(self.augmentations)), size=self.size, replace=False, p=self.p)

        outputs = inputs
        for i in indices:
            outputs = self.augmentations[i](outputs)

        return outputs


class FixedStrongAugmenter(BaseAugmenter):
    def __init__(self, image_size: ImageSizeType, dataset_mean: DatasetStatType, dataset_std: DatasetStatType,
                 padding: PaddingInputType = 0.125, pad_if_needed: bool = False):
        """

        :param image_size: (height, width) image size
        :param dataset_mean: dataset mean value for each channel in CHW
        :param dataset_std: dataset standard deviation for each channel in CHW
        """
        super().__init__(image_size, dataset_mean, dataset_std, padding, pad_if_needed)

        self.transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.2),
            K.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), ratio=(1., 1.)),
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
            K.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        )

    def forward(self, images: Tensor) -> Tensor:
        out = self.transform(images)
        return out


class SimpleAugmenter(BaseAugmenter):
    def __init__(self, image_size: ImageSizeType, dataset_mean: DatasetStatType, dataset_std: DatasetStatType,
                 padding: PaddingInputType = 0.125, pad_if_needed: bool = False):
        """

        :param image_size: (height, width) image size
        :param dataset_mean: dataset mean value for each channel in CHW
        :param dataset_std: dataset standard deviation for each channel in CHW
        :param pad_if_needed: bool flag for RandomCrop "pad_if_needed" option
        """
        super().__init__(image_size, dataset_mean, dataset_std, padding, pad_if_needed)

        self.transform = nn.Sequential(
            K.RandomCrop(size=self.image_size, padding=self.padding, pad_if_needed=self.pad_if_needed,
                         padding_mode='reflect'),
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        )

    def forward(self, images: Tensor) -> Tensor:
        out = self.transform(images)
        return out


class RandAugmentAugmenter(BaseAugmenter):
    def __init__(self, image_size: ImageSizeType, dataset_mean: DatasetStatType, dataset_std: DatasetStatType,
                 padding: PaddingInputType = 0.125, pad_if_needed: bool = False, subset_size: int = 2):
        """

        :param image_size: (height, width) image size
        :param dataset_mean: dataset mean value for each channel in CHW
        :param dataset_std: dataset standard deviation for each channel in CHW
        :param subset_size: number of augmentations used in subset
        """
        super().__init__(image_size, dataset_mean, dataset_std, padding, pad_if_needed)

        from .randaugment import RandAugNS

        self.transform = nn.Sequential(
            K.RandomCrop(size=self.image_size, padding=self.padding, pad_if_needed=self.pad_if_needed,
                         padding_mode='reflect'),
            K.RandomHorizontalFlip(p=0.5),
            RandAugNS(n=subset_size, m=10),
            K.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        )

    def forward(self, images: Tensor) -> Tensor:
        outputs = self.transform(images)

        return outputs
