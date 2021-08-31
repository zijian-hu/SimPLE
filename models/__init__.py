import torch

from functools import partial

from .augmentation import get_augmenter
from .rampup import RampUp, LinearRampUp, get_ramp_up
from .utils import (get_accuracy, interleave, unwrap_model, split_classifier_params)
from .models import EMA, build_model, build_ema_model
from .optimization import build_optimizer, build_lr_scheduler

from . import augmentation
from . import rampup
from . import utils
from . import mixmatch
from . import optimization
from . import types
from . import models

# for type hint
from typing import Optional, Union, Set
from argparse import Namespace
from torch.nn import Module

from .types import OptimizerParametersType, MixMatchFunctionType


def load_pretrain(model: Module,
                  checkpoint_path: str,
                  allowed_prefix: str,
                  ignored_prefix: str,
                  device: torch.device,
                  checkpoint_key: Optional[str] = None):
    full_allowed_prefix = f"{allowed_prefix}." if bool(allowed_prefix) else allowed_prefix
    full_ignored_prefix = f"{ignored_prefix}." if bool(ignored_prefix) else ignored_prefix

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint_key is not None:
        pretrain_state_dict = checkpoint[checkpoint_key]
    else:
        pretrain_state_dict = checkpoint

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


def get_mixmatch_function(args: Namespace,
                          num_classes: int,
                          augmenter: Module,
                          strong_augmenter: Module) -> MixMatchFunctionType:
    if args.mixmatch_type == "simple":
        from .mixmatch import SimPLE

        return SimPLE(augmenter=augmenter,
                      strong_augmenter=strong_augmenter,
                      num_classes=num_classes,
                      temperature=args.t,
                      num_augmentations=args.k,
                      num_strong_augmentations=args.k_strong,
                      is_strong_augment_x=False,
                      train_label_guessing=False)

    elif args.mixmatch_type == "enhanced":
        from .mixmatch import MixMatchEnhanced

        return MixMatchEnhanced(augmenter=augmenter,
                                strong_augmenter=strong_augmenter,
                                num_classes=num_classes,
                                temperature=args.t,
                                num_augmentations=args.k,
                                num_strong_augmentations=args.k_strong,
                                alpha=args.alpha,
                                is_strong_augment_x=False,
                                train_label_guessing=False)

    elif args.mixmatch_type == "mixmatch":
        from .mixmatch import MixMatch

        return MixMatch(augmenter=augmenter,
                        num_classes=num_classes,
                        temperature=args.t,
                        num_augmentations=args.k,
                        alpha=args.alpha,
                        train_label_guessing=False)

    else:
        raise NotImplementedError(f"{args.mixmatch_type} is not a supported mixmatch type")


__all__ = [
    # modules
    "augmentation",
    "rampup",
    "utils",
    "mixmatch",
    "models",
    "optimization",
    "types",

    # classes
    "EMA",

    # functions
    "interleave",
    "get_augmenter",
    "get_ramp_up",
    "get_accuracy",
    "build_model",
    "build_ema_model",
    "build_optimizer",
    "build_lr_scheduler",
    "load_pretrain",
    "get_trainable_params",
    "get_mixmatch_function",
]
