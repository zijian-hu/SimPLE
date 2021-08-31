import warnings

from .wide_resnet import WideResNet
from .resnet import *
from .ema import EMA, FullEMA
from .utils import unwrap_model

from . import utils

# for type hint
from typing import Union
from torch.nn import Module


def get_model_size(model: Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(model_type: str, in_channels: int, out_channels: int, **kwargs) -> Union[Module, EMA]:
    if model_type == "wrn28-8":
        model = WideResNet(in_channels=in_channels,
                           out_channels=out_channels,
                           depth=28,
                           widening_factor=8,
                           base_channels=16,
                           **kwargs)

    elif model_type == "resnet18":
        model = resnet18(pretrained=False, progress=True, num_classes=out_channels, **kwargs)

    elif model_type == "resnet50":
        model = resnet50(pretrained=False, progress=True, num_classes=out_channels, **kwargs)

    elif model_type == "wrn28-2":
        # default model is WRN 28-2
        model = WideResNet(in_channels=in_channels,
                           out_channels=out_channels,
                           depth=28,
                           widening_factor=2,
                           base_channels=16,
                           **kwargs)

    else:
        raise NotImplementedError(f"\"{model_type}\" is not a supported model type")

    print(f'{model_type} Total params: {(get_model_size(model) / 1e6):.2f}M')
    return model


def build_ema_model(model: Module, ema_type: str, ema_decay: float) -> Union[Module, EMA]:
    if ema_decay == 0:
        warnings.warn("EMA decay is 0, turn off EMA")
        return model

    elif ema_type == "full":
        return FullEMA(model, decay=ema_decay)

    elif ema_type == "default":
        return EMA(model, decay=ema_decay)

    else:
        raise NotImplementedError(f"\"{ema_type}\" is not a supported EMA type ")


__all__ = [
    # modules,
    "utils",

    # classes
    "WideResNet",
    "ResNet",
    "EMA",
    "FullEMA",

    # functions
    "get_model_size",
    "build_model",
    "build_ema_model",

    # model functions
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]
