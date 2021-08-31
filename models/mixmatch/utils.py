import torch
import numpy as np
from torch.nn import functional as F

from functools import reduce

from ..utils import set_model_mode

# for type hint
from typing import Sequence, Tuple
from torch import Tensor
from torch.nn import Module


def label_guessing(batches: Sequence[Tensor], model: Module, is_train_mode: bool = True) -> Tensor:
    with set_model_mode(model, is_train_mode):
        with torch.no_grad():
            probs = [F.softmax(model(batch), dim=1) for batch in batches]
            mean_prob = reduce(lambda x, y: x + y, probs) / len(batches)

    return mean_prob


def sharpen(x: Tensor, temperature: float) -> Tensor:
    sharpened_x = x ** (1 / temperature)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


def mixup(x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, alpha: float) -> Tuple[Tensor, Tensor]:
    # lambda is a reserved word in python, substituting by beta
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
