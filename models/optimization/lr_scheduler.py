import numpy as np
from torch.optim.lr_scheduler import LambdaLR, StepLR

# for type hint
from torch.optim.optimizer import Optimizer

from .types import LRSchedulerType


class CosineDecay:
    def __init__(self, max_iter: int, factor: float, min_value: float = 0., max_value: float = 1.):
        self.max_iter = max_iter
        self.factor = factor

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, curr_step: int) -> float:
        output = np.cos(self.factor * np.pi * min(curr_step, self.max_iter) / self.max_iter)

        return np.clip(output, self.min_value, self.max_value).item()


def build_lr_scheduler(scheduler_type: str,
                       optimizer: Optimizer,
                       max_iter: int,
                       cosine_factor: float,
                       step_size: int,
                       gamma: float) -> LRSchedulerType:
    if scheduler_type == "cosine_decay":
        return LambdaLR(optimizer=optimizer,
                        lr_lambda=CosineDecay(max_iter=max_iter, factor=cosine_factor))

    if scheduler_type == "step_decay":
        return StepLR(optimizer=optimizer,
                      step_size=step_size,
                      gamma=gamma)

    else:
        # dummy scheduler
        return LambdaLR(optimizer=optimizer, lr_lambda=lambda curr_iter: 1.0)


__all__ = [
    # functions
    "build_lr_scheduler",

    # classes
    "CosineDecay",
]
