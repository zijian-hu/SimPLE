import torch
from kornia import filters as F
import numpy as np

from random import random

# for type hint
from typing import Optional, Tuple, Sequence
from torch import Tensor
from torch.nn import Module


class RandomAugmentation(Module):
    def __init__(self, augmentation: Module, p: float = 0.5, same_on_batch: bool = False):
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


class RandomGaussianBlur(Module):
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


class RandomChoice(Module):
    def __init__(self, augmentations: Sequence[Module], size: int = 2, p: Optional[Sequence[float]] = None):
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
