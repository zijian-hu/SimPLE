import torch
from torch import nn
from torch import optim

from functools import partial

from .ema import EMA, FullEMA

from .wide_resnet import WideResNet

from .augmentation import (BaseAugmenter, get_augmenter)
from .rampup import RampUp, LinearRampUp
from .utils import (interleave, unwrap_model, split_classifier_params, consume_prefix_in_state_dict_if_present)
from .optimization import build_lr_scheduler

from . import augmentation
from . import rampup
from . import utils
from . import mixmatch
from . import optimization
from . import types

# for type hint
from typing import Optional, Union, Set
from argparse import Namespace
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .types import OptimizerParametersType, MixMatchFunctionType


def _get_model_size(model: Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(model_type: str,
                in_channels: int,
                out_channels: int,
                use_ema: bool,
                ema_type: str,
                ema_decay: float,
                **kwargs) -> Union[Module, EMA]:
    if model_type == "wrn28-8":
        model = WideResNet(in_channels=in_channels,
                           out_channels=out_channels,
                           depth=28,
                           widening_factor=8,
                           base_channels=16,
                           **kwargs)

    elif model_type == "resnet18":
        from .resnet import resnet18
        model = resnet18(pretrained=False, progress=True, num_classes=out_channels, **kwargs)

    elif model_type == "resnet50":
        from .resnet import resnet50
        model = resnet50(pretrained=False, progress=True, num_classes=out_channels, **kwargs)

    else:
        # default model is WRN 28-2
        model = WideResNet(in_channels=in_channels,
                           out_channels=out_channels,
                           depth=28,
                           widening_factor=2,
                           base_channels=16,
                           **kwargs)

    print(f'{model_type} Total params: {(_get_model_size(model) / 1e6):.2f}M')

    if use_ema:
        assert ema_decay is not None
        if ema_type == "full":
            model = FullEMA(model, decay=ema_decay)
        else:
            model = EMA(model, decay=ema_decay)

    return model


def get_trainable_params(model: Module,
                         learning_rate: float,
                         feature_learning_rate: Optional[float],
                         classifier_prefix: Union[str, Set[str]] = 'fc',
                         requires_grad_only: bool = True) -> OptimizerParametersType:
    if feature_learning_rate is not None:
        embedder_weights, classifier_weights = split_classifier_params(model, classifier_prefix)

        if requires_grad_only:
            # keep only the parameters that requires grad
            embedder_weights = [param for param in embedder_weights if param.requires_grad]
            classifier_weights = [param for param in classifier_weights if param.requires_grad]

        params = [dict(params=embedder_weights, lr=feature_learning_rate),
                  dict(params=classifier_weights, lr=learning_rate)]
    else:
        params = model.parameters()

        if requires_grad_only:
            # keep only the parameters that requires grad
            params = [param for param in params if param.requires_grad]

    return params


def build_optimizer(optimizer_type: str,
                    params: OptimizerParametersType,
                    learning_rate: float,
                    weight_decay: float,
                    momentum: float) -> Optimizer:
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params,
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              nesterov=True)
    else:
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def get_mixmatch_function(args: Namespace,
                          model: Union[Module, EMA],
                          num_classes: int,
                          augmenter: Module,
                          strong_augmenter: Module) -> MixMatchFunctionType:
    # TODO: handle distributed model
    if isinstance(model, EMA) and not args.ema_label_guessing:
        output_model = model.model
    else:
        output_model = model

    if args.mixmatch_type == "simple":
        return partial(
            mixmatch.simple_mixmatch,
            model=output_model,
            augmenter=augmenter,
            strong_augmenter=strong_augmenter,
            num_classes=num_classes,
            t=args.t,
            k=args.k,
            k_strong=args.k_strong)

    elif args.mixmatch_type == "enhanced":
        return partial(
            mixmatch.mixmatch_enhanced,
            model=output_model,
            augmenter=augmenter,
            strong_augmenter=strong_augmenter,
            num_classes=num_classes,
            t=args.t,
            k=args.k,
            k_strong=args.k_strong,
            alpha=args.alpha)
    else:
        return partial(
            mixmatch.mixmatch,
            model=output_model,
            augmenter=augmenter,
            num_classes=num_classes,
            t=args.t,
            k=args.k,
            alpha=args.alpha)


def load_pretrain(model: nn.Module,
                  checkpoint_path: str,
                  allowed_prefix: str,
                  ignored_prefix: str,
                  device: torch.device,
                  checkpoint_key: Optional[str] = "params") -> None:
    full_allowed_prefix = f"{allowed_prefix}." if bool(allowed_prefix) else allowed_prefix
    full_ignored_prefix = f"{ignored_prefix}." if bool(ignored_prefix) else ignored_prefix

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint_key is not None:
        pretrain_state_dict = checkpoint[checkpoint_key]
    else:
        pretrain_state_dict = checkpoint

    # remove DP/DDP wrapper
    consume_prefix_in_state_dict_if_present(pretrain_state_dict, prefix="module.")

    shadow = None
    if isinstance(model, EMA):
        shadow = unwrap_model(model.shadow)
        model = unwrap_model(model.model)

    state_dict = model.state_dict()

    for name, param in pretrain_state_dict.items():
        if name.startswith(full_allowed_prefix) and not name.startswith(full_ignored_prefix):
            name = name[len(full_allowed_prefix):]

            assert name in state_dict.keys()
            state_dict[name] = param

    # load pretrain model
    model.load_state_dict(state_dict)
    if shadow is not None:
        shadow.load_state_dict(state_dict)


__all__ = [
    # modules
    "augmentation",
    "rampup",
    "utils",
    "mixmatch",
    "optimization",
    "types",

    # classes
    "EMA",
    "WideResNet",
    "LinearRampUp",

    # functions
    "interleave",
    "get_augmenter",
    "build_model",
    "get_trainable_params",
    "build_optimizer",
    "get_mixmatch_function",
    "load_pretrain",
    "build_lr_scheduler",
]
