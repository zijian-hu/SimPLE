import torch
from torch import nn
import numpy as np

from .augmenter import (BaseAugmenter, SimpleAugmenter, FixedStrongAugmenter, RandAugmentAugmenter)

# for type hint
from torch import Tensor
from typing import Tuple, Sequence, Union


def get_augmenter(augmenter_type: str,
                  image_size: Union[Tuple[int, int], np.ndarray],
                  dataset_mean: Union[Sequence[float], Tensor],
                  dataset_std: Union[Sequence[float], Tensor],
                  **augmenter_kwargs) -> Union[BaseAugmenter, nn.Module]:
    """

    :param augmenter_type:
    :param image_size:
    :param dataset_mean: dataset mean value in CHW
    :param dataset_std: dataset mean value in CHW
    :param augmenter_kwargs:
    :return:
    """
    if isinstance(image_size, np.ndarray):
        image_size = tuple(image_size.tolist())[:2]

    if not isinstance(dataset_mean, Tensor):
        dataset_mean = torch.tensor(dataset_mean, dtype=torch.float32)
    if not isinstance(dataset_std, Tensor):
        dataset_std = torch.tensor(dataset_std, dtype=torch.float32)

    augmenter_type = augmenter_type.strip().lower()

    if augmenter_type == "simple":
        return SimpleAugmenter(image_size=image_size, dataset_mean=dataset_mean, dataset_std=dataset_std,
                               **augmenter_kwargs)
    elif augmenter_type == "fixed":
        return FixedStrongAugmenter(image_size=image_size, dataset_mean=dataset_mean, dataset_std=dataset_std)

    elif augmenter_type in ["validation", "test"]:
        from kornia import augmentation as K

        return nn.Sequential(
            K.Normalize(mean=dataset_mean, std=dataset_std),
        )

    elif augmenter_type == "randaugment":
        return RandAugmentAugmenter(image_size=image_size, dataset_mean=dataset_mean, dataset_std=dataset_std)

    else:
        raise ValueError(f"\"{augmenter_type}\" is not a supported augmenter type")


__all__ = [
    # modules

    # classes
    "BaseAugmenter",
    "SimpleAugmenter",
    "FixedStrongAugmenter",
    "RandAugmentAugmenter",

    # functions
    "get_augmenter",
]
