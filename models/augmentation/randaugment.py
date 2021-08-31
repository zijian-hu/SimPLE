import torch
import random
import numpy as np
from torch.nn import functional as F
from kornia.geometry import transform as T
from kornia import enhance as E
from kornia.augmentation import functional as KF

# for type hint
from typing import Any, Optional, List, Tuple, Callable
from torch import Tensor
from torch.nn import Module


# Affine
def translate_x(x: Tensor, v: float) -> Tensor:
    B, C, H, W = x.shape
    return T.translate(x, torch.tensor([[v * W, 0]], device=x.device, dtype=x.dtype))


def translate_y(x: Tensor, v: float) -> Tensor:
    B, C, H, W = x.shape
    return T.translate(x, torch.tensor([[0, v * H]], device=x.device, dtype=x.dtype))


def shear_x(x: Tensor, v: float) -> Tensor:
    return T.shear(x, torch.tensor([[v, 0.0]], device=x.device, dtype=x.dtype))


def shear_y(x: Tensor, v: float) -> Tensor:
    return T.shear(x, torch.tensor([[0.0, v]], device=x.device, dtype=x.dtype))


def rotate(x: Tensor, v: float) -> Tensor:
    return T.rotate(x, torch.tensor([v], device=x.device, dtype=x.dtype))


def auto_contrast(x: Tensor, _: Any) -> Tensor:
    B, C, H, W = x.shape

    x_min = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out = (x - x_min) / torch.clamp(x_max - x_min, min=1e-9, max=1)

    return x_out.expand_as(x)


def invert(x: Tensor, _: Any) -> Tensor:
    return 1.0 - x


def equalize(x: Tensor, _: Any) -> Tensor:
    return KF.apply_equalize(x, params=dict(batch_prob=torch.tensor([1.0] * len(x), dtype=x.dtype, device=x.device)))


def flip(x: Tensor, _: Any) -> Tensor:
    return T.hflip(x)


def solarize(x: Tensor, v: float) -> Tensor:
    x[x < v] = 1 - x[x < v]
    return x


def brightness(x: Tensor, v: float) -> Tensor:
    return E.adjust_brightness(x, v)


def color(x: Tensor, v: float) -> Tensor:
    return E.adjust_saturation(x, v)


def contrast(x: Tensor, v: float) -> Tensor:
    return E.adjust_contrast(x, v)


def sharpness(x: Tensor, v: float) -> Tensor:
    return KF.apply_sharpness(x, params=dict(sharpness_factor=v))


def identity(x: Tensor, _: Any) -> Tensor:
    return x


def posterize(x: Tensor, v: float) -> Tensor:
    v = int(v)
    return E.posterize(x, v)


def cutout(x: Tensor, v: float) -> Tensor:
    B, C, H, W = x.shape

    x_v = int(v * W)
    y_v = int(v * H)

    x_idx = np.random.uniform(low=0, high=W - x_v, size=(B, 1, 1, 1)) + np.arange(x_v).reshape((1, 1, 1, -1))
    y_idx = np.random.uniform(low=0, high=H - y_v, size=(B, 1, 1, 1)) + np.arange(y_v).reshape((1, 1, -1, 1))

    x[np.arange(B).reshape((B, 1, 1, 1)), np.arange(C).reshape((1, C, 1, 1)), y_idx, x_idx] = 0.5
    return x


def cutout_pad(x: Tensor, v: float) -> Tensor:
    B, C, H, W = x.shape

    x = F.pad(x, [int(v * W / 2), int(v * W / 2), int(v * H / 2), int(v * H / 2)])

    x = cutout(x, v / (1 + v))

    x = T.center_crop(x, (H, W))

    return x


class RandAugment(Module):
    def __init__(self, n: int, m: int, augmentation_pool: Optional[List[Tuple[Callable, float, float]]] = None):
        """
        
        Args:
            n: number of transformations 
            m: magnitude
            augmentation_pool: transformation pool
        """
        super().__init__()

        self.n = n
        self.m = m
        if augmentation_pool is not None:
            self.augmentation_pool = augmentation_pool
        else:
            self.augmentation_pool = [
                (auto_contrast, np.nan, np.nan),
                (brightness, 0.05, 0.95),
                (color, 0.05, 0.95),
                (contrast, 0.05, 0.95),
                (cutout, 0, 0.3),
                (equalize, np.nan, np.nan),
                (identity, np.nan, np.nan),
                (posterize, 4, 8),
                (rotate, -30, 30),
                (sharpness, 0.05, 0.95),
                (shear_x, -0.3, 0.3),
                (shear_y, -0.3, 0.3),
                (solarize, 0.0, 1.0),
                (translate_x, -0.3, 0.3),
                (translate_y, -0.3, 0.3),
            ]

    def forward(self, x):
        assert len(x.shape) == 4

        ops = random.choices(self.augmentation_pool, k=self.n)

        for op, min_v, max_v in ops:
            v = random.randint(1, self.m + 1) / 10 * (max_v - min_v) + min_v
            x = op(x, v)

        return x


class RandAugmentNS(RandAugment):
    def __init__(self, n: int, m: int):
        super(RandAugmentNS, self).__init__(
            n=n,
            m=m,
            augmentation_pool=[
                (auto_contrast, np.nan, np.nan),
                (brightness, 0.05, 0.95),
                (color, 0.05, 0.95),
                (contrast, 0.05, 0.95),
                (equalize, np.nan, np.nan),
                (identity, np.nan, np.nan),
                (posterize, 4, 8),
                (rotate, -30, 30),
                (sharpness, 0.05, 0.95),
                (shear_x, -0.3, 0.3),
                (shear_y, -0.3, 0.3),
                (solarize, 0.0, 1.0),
                (translate_x, -0.3, 0.3),
                (translate_y, -0.3, 0.3),
            ])

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4

        ops = random.choices(self.augmentation_pool, k=self.n)

        for op, min_v, max_v in ops:
            v = random.randint(1, self.m + 1) / 10 * (max_v - min_v) + min_v
            if random.random() < 0.5:
                x = op(x, v)

        x = cutout(x, 0.5)
        return x
