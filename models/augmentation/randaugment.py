"""
References:
https://github.com/ildoonet/pytorch-randaugment/tree/master/RandAugment
https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
"""
from torch import nn
import torch.nn.functional as F
import kornia.geometry.transform as T
import kornia.augmentation
import kornia.enhance as E
import torch
import random
import numpy as np
from kornia.enhance.adjust import _to_bchw

# for type hint
from typing import Union
from torch import Tensor


def kornia_sharpness(inputs: Tensor, factor: Union[float, Tensor]) -> Tensor:
    r"""Implements Sharpness function from PIL using torch ops.
    Args:
        inputs (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to sharpen.
        factor (float or torch.Tensor): factor of sharpness strength. Must be above 0.
            If float or one element tensor, input will be sharpened by the same factor across the whole batch.
            If 1-d tensor, input will be sharpened element-wisely, len(factor) == len(input).
    Returns:
        torch.Tensor: Sharpened image or images.
    """
    inputs = _to_bchw(inputs)
    if isinstance(factor, Tensor):
        factor = factor.squeeze()
        if len(factor.size()) != 0:
            assert inputs.size(0) == factor.size(0), \
                f"Input batch size shall match with factor size if 1d array. Got {inputs.size(0)} and {factor.size(0)}"
    else:
        factor = float(factor)
    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ], dtype=inputs.dtype, device=inputs.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2
    degenerate = torch.nn.functional.conv2d(inputs, kernel, bias=None, stride=1, groups=inputs.size(1))
    degenerate = torch.clamp(degenerate, 0., 1.)

    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, inputs)

    def _blend_one(input1: Tensor, input2: Tensor, factor: Union[float, Tensor]) -> Tensor:
        if isinstance(factor, Tensor):
            factor = factor.squeeze()
            assert len(factor.size()) == 0, f"Factor shall be a float or single element tensor. Got {factor}"
        if factor == 0.:
            return input1
        if factor == 1.:
            return input2
        diff = (input2 - input1) * factor
        res = input1 + diff
        if factor > 0. and factor < 1.:
            return res
        return torch.clamp(res, 0, 1)

    if isinstance(factor, (float)) or len(factor.size()) == 0:
        return _blend_one(inputs, result, factor)
    return torch.stack([_blend_one(inputs[i], result[i], factor[i]) for i in range(len(factor))])


def kornia_equalize(inputs: Tensor) -> Tensor:
    """Implements Equalize function from PIL using PyTorch ops based on uint8 format:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352
    Args:
        inputs (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to equalize.
    Returns:
        torch.Tensor: Sharpened image or images.
    """
    inputs = _to_bchw(inputs) * 255

    # Code taken from: https://github.com/pytorch/vision/pull/796
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[c, :, :]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1, device=inputs.device), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)

        return result / 255.

    res = []
    for image in inputs:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_image = torch.stack([scale_channel(image, i) for i in range(len(image))])
        res.append(scaled_image)
    return torch.stack(res)


# Affine
def translate_x(x, v):
    B, C, H, W = x.shape
    # TODO: not sure about x direction is width
    return T.translate(x, torch.tensor([[v * W, 0]], device=x.device, dtype=x.dtype))


def translate_y(x, v):
    B, C, H, W = x.shape
    # TODO: not sure about y direction is height
    return T.translate(x, torch.tensor([[0, v * H]], device=x.device, dtype=x.dtype))


def shear_x(x, v):
    return T.shear(x, torch.tensor([[v, 0.0]], device=x.device, dtype=x.dtype))


def shear_y(x, v):
    return T.shear(x, torch.tensor([[0.0, v]], device=x.device, dtype=x.dtype))


def rotate(x, v):
    return T.rotate(x, torch.tensor([v], device=x.device, dtype=x.dtype))


# TODO: not sure
def auto_contrast(x, _):
    B, C, H, W = x.shape

    x_min = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out = (x - x_min) / torch.clamp(x_max - x_min, min=1e-9, max=1)

    return x_out.expand_as(x)


# TODO: not sure
def invert(x, _):
    return 1.0 - x


def equalize(x, _):
    return kornia_equalize(x)


def flip(x, _):
    return T.hflip(x)


def solarize(x, v):
    x[x < v] = 1 - x[x < v]
    return x


# TODO: not like pil
def brightness(x, v):
    return E.adjust_brightness(x, v)


# TODO: not like pil
def color(x, v):
    return E.adjust_saturation(x, v)


# TODO: not like pil
def contrast(x, v):
    return E.adjust_contrast(x, v)


# TODO: not like pil
def sharpness(x, v):
    return kornia_sharpness(x, v)


def identity(x, _):
    return x


def posterize(x, v):
    v = int(v)
    return E.posterize(x, v)


def cutout(x, v):
    B, C, H, W = x.shape

    x_v = int(v * W)
    y_v = int(v * H)

    x_idx = np.random.uniform(low=0, high=W - x_v, size=(B, 1, 1, 1)) + np.arange(x_v).reshape((1, 1, 1, -1))
    y_idx = np.random.uniform(low=0, high=H - y_v, size=(B, 1, 1, 1)) + np.arange(y_v).reshape((1, 1, -1, 1))

    x[np.arange(B).reshape((B, 1, 1, 1)), np.arange(C).reshape((1, C, 1, 1)), y_idx, x_idx] = 0.5
    return x


def cutout_pad(x, v):
    B, C, H, W = x.shape

    x = F.pad(x, [int(v * W / 2), int(v * W / 2), int(v * H / 2), int(v * H / 2)])

    x = cutout(x, v / (1 + v))

    x = T.center_crop(x, (H, W))

    return x


class RandAug(nn.Module):
    def __init__(self, n, m, aug_pool=None):
        super().__init__()
        self.n = n
        self.m = m
        if aug_pool is not None:
            self.aug_pool = aug_pool
        else:
            self.aug_pool = [
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

        ops = random.choices(self.aug_pool, k=self.n)

        for op, min_v, max_v in ops:
            v = random.randint(1, self.m + 1) / 10 * (max_v - min_v) + min_v
            x = op(x, v)

        return x


class RandAugNS(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m
        self.aug_pool = [
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
        ]

    def forward(self, x):
        assert len(x.shape) == 4

        ops = random.choices(self.aug_pool, k=self.n)

        for op, min_v, max_v in ops:
            v = random.randint(1, self.m + 1) / 10 * (max_v - min_v) + min_v
            if random.random() < 0.5:
                x = op(x, v)

        x = cutout(x, 0.5)
        return x
