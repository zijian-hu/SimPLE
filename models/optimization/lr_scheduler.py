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
        max_iter = max(self.max_iter, 1)
        curr_step = np.clip(curr_step, 0, max_iter).item()

        return self.compute_cosine_decay(curr_step=curr_step,
                                         max_iter=max_iter,
                                         factor=self.factor,
                                         min_value=self.min_value,
                                         max_value=self.max_value)

    @staticmethod
    def compute_cosine_decay(curr_step: int,
                             max_iter: int,
                             factor: float,
                             min_value: float,
                             max_value: float) -> float:
        output = np.cos(factor * np.pi * float(curr_step) / float(max(max_iter, 1)))

        return np.clip(output, min_value, max_value).item()


class CosineWarmupDecay(CosineDecay):
    def __init__(self,
                 max_iter: int,
                 factor: float,
                 num_warmup_steps: int,
                 min_value: float = 0.,
                 max_value: float = 1.):
        super(CosineWarmupDecay, self).__init__(max_iter=max_iter,
                                                factor=factor,
                                                min_value=min_value,
                                                max_value=max_value)

        self.num_warmup_steps = max(num_warmup_steps, 0)

    def __call__(self, curr_step: int) -> float:
        if curr_step < self.num_warmup_steps:
            return float(curr_step) / float(max(self.num_warmup_steps, 1))
        else:
            max_iter = max(self.max_iter - self.num_warmup_steps, 1)
            curr_step = np.clip(curr_step - self.num_warmup_steps, 0, max_iter).item()

            return self.compute_cosine_decay(curr_step=curr_step,
                                             max_iter=max_iter,
                                             factor=self.factor,
                                             min_value=self.min_value,
                                             max_value=self.max_value)


def build_lr_scheduler(scheduler_type: str,
                       optimizer: Optimizer,
                       max_iter: int,
                       cosine_factor: float,
                       step_size: int,
                       gamma: float,
                       num_warmup_steps: int,
                       **kwargs) -> LRSchedulerType:
    if scheduler_type == "cosine_decay":
        return LambdaLR(optimizer=optimizer,
                        lr_lambda=CosineDecay(max_iter=max_iter, factor=cosine_factor))

    elif scheduler_type == "cosine_warmup_decay":
        return LambdaLR(optimizer=optimizer,
                        lr_lambda=CosineWarmupDecay(max_iter=max_iter,
                                                    factor=cosine_factor,
                                                    num_warmup_steps=num_warmup_steps))

    elif scheduler_type == "step_decay":
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
