import torch
from torch import nn
import numpy as np

from .utils import label_guessing, sharpen
from ..utils import to_one_hot

# for type hint
from typing import Tuple
from torch import Tensor


@torch.no_grad()
def mixmatch_enhanced(x_inputs: Tensor,
                      x_targets: Tensor,
                      u_inputs: Tensor,
                      u_true_targets: Tensor,
                      model: nn.Module,
                      augmenter: nn.Module,
                      strong_augmenter: nn.Module,
                      num_classes: int,
                      t: float = 0.5,
                      k: int = 2,
                      k_strong: int = 2,
                      alpha: float = 0.75) -> Tuple[Tensor, ...]:
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
    u_augmented = torch.cat(u_weak_aug + u_strong_aug, dim=0)
    q_guess = torch.cat([q_guess for _ in range(k + k_strong)], dim=0)
    q_true = torch.cat([u_true_targets_one_hot for _ in range(k + k_strong)], dim=0)
    assert len(u_augmented) == len(q_guess) == len(q_true)

    # random shuffle according to the paper
    indices = list(range(len(x_augmented) + len(u_augmented)))
    np.random.shuffle(indices)

    # MixUp
    wx = torch.cat([x_augmented, u_augmented], dim=0)
    wy = torch.cat([x_targets_one_hot, q_guess], dim=0)
    wq = torch.cat([x_targets_one_hot, q_true], dim=0)
    assert len(wx) == len(wy) == len(wq)
    assert len(wx) == len(x_augmented) + len(u_augmented)
    assert len(wy) == len(x_targets_one_hot) + len(q_guess)
    wx_shuffled = wx[indices]
    wy_shuffled = wy[indices]
    wq_shuffled = wq[indices]
    assert len(wx) == len(wx_shuffled)
    assert len(wy) == len(wy_shuffled)
    assert len(wq) == len(wq_shuffled)

    # the official version use the same lambda ~ sampled from Beta(alpha, alpha) for both labeled and unlabeled inputs
    # wx_mixed, wy_mixed = mixup(x1=wx, x2=wx_shuffled, y1=wy, y2=wy_shuffled, alpha=alpha)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    wx_mixed = lam * wx + (1 - lam) * wx_shuffled
    wy_mixed = lam * wy + (1 - lam) * wy_shuffled
    wq_mixed = lam * wq + (1 - lam) * wq_shuffled

    x_mixed, p_mixed = wx_mixed[:len(x_augmented)], wy_mixed[:len(x_augmented)]
    u_mixed, q_mixed = wx_mixed[len(x_augmented):], wy_mixed[len(x_augmented):]
    q_true_mixed = wq_mixed[len(x_augmented):]

    return x_mixed, p_mixed, u_mixed, q_mixed, q_true_mixed
