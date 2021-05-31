from torch.optim.lr_scheduler import LambdaLR, StepLR

from typing import Union

LRSchedulerType = Union[LambdaLR, StepLR]

__all__ = [
    "LRSchedulerType",
]
