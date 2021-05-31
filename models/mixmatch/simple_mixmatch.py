from typing import Tuple

import torch
from torch import Tensor, nn

from .utils import label_guessing, sharpen
from ..utils import to_one_hot


@torch.no_grad()
def simple_mixmatch(x_inputs: Tensor,
                    x_targets: Tensor,
                    u_inputs: Tensor,
                    u_true_targets: Tensor,
                    model: nn.Module,
                    augmenter: nn.Module,
                    strong_augmenter: nn.Module,
                    num_classes: int,
                    t: float = 0.5,
                    k: int = 2,
                    k_strong: int = 2) -> Tuple[Tensor, ...]:
    # convert targets to one-hot
    x_targets_one_hot = to_one_hot(x_targets, num_classes=num_classes, dtype=x_inputs.dtype)
    u_true_targets_one_hot = to_one_hot(u_true_targets, num_classes=num_classes, dtype=x_inputs.dtype)

    # apply augmentations
    x_augmented = augmenter(x_inputs)

    u_weak_aug = [augmenter(u_inputs) for _ in range(k)]
    u_strong_aug = [strong_augmenter(u_inputs) for _ in range(k_strong)]

    # label guessing with weakly augmented data
    q_guess = label_guessing(model, u_weak_aug, is_train_mode=False)
    q_guess = sharpen(q_guess, t)

    # concat the unlabeled data and targets
    u_augmented = torch.cat(u_strong_aug, dim=0)
    q_guess = torch.cat([q_guess for _ in range(k_strong)], dim=0)
    q_true = torch.cat([u_true_targets_one_hot for _ in range(k_strong)], dim=0)
    assert len(u_augmented) == len(q_guess) == len(q_true)

    return x_augmented, x_targets_one_hot, u_augmented, q_guess, q_true
