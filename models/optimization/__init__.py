from torch.optim import SGD, AdamW

from .lr_scheduler import build_lr_scheduler

from . import lr_scheduler
from . import types

# for type hint
from torch.optim.optimizer import Optimizer

from .types import OptimizerParametersType


def build_optimizer(optimizer_type: str,
                    params: OptimizerParametersType,
                    learning_rate: float,
                    weight_decay: float,
                    momentum: float) -> Optimizer:
    if optimizer_type == "sgd":
        optimizer = SGD(params,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        nesterov=True)

    elif optimizer_type == "adamw":
        optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    else:
        raise NotImplementedError(f"\"{optimizer_type}\" is not a supported optimizer type")

    return optimizer


__all__ = [
    # classes
    # modules
    "lr_scheduler",
    "types",

    # functions
    "build_optimizer",
    "build_lr_scheduler",
]
