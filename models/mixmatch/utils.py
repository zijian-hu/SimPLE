from functools import reduce
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..utils import set_model_mode


def label_guessing(model: nn.Module, batches: Sequence[Tensor], is_train_mode: bool = True) -> Tensor:
    with set_model_mode(model, is_train_mode):
        with torch.no_grad():
            probs = [F.softmax(model(batch), dim=1) for batch in batches]
            mean_prob = reduce(lambda x, y: x + y, probs) / len(batches)

    return mean_prob


def sharpen(x: Tensor, t: float) -> Tensor:
    sharpened_x = x ** (1 / t)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


def mixup(x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, alpha: float) -> Tuple[Tensor, Tensor]:
    # lambda is a reserved word in python, substituting by beta
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
