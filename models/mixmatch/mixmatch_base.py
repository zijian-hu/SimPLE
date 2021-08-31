"""
code inspired by https://github.com/gan3sh500/mixmatch-pytorch and
https://github.com/google-research/mixmatch
"""
import torch
import numpy as np
from torch.nn import functional as F

from .utils import label_guessing, sharpen

# for type hint
from typing import Optional, Dict, Union, List, Sequence
from torch import Tensor
from torch.nn import Module


class MixMatchBase:
    def __init__(self,
                 augmenter: Module,
                 strong_augmenter: Optional[Module],
                 num_classes: int,
                 temperature: float,
                 num_augmentations: int,
                 num_strong_augmentations: int,
                 alpha: float,
                 is_strong_augment_x: bool,
                 train_label_guessing: bool):
        # callables
        self.augmenter = augmenter
        self.strong_augmenter = strong_augmenter

        # parameters
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha

        self.num_augmentations = num_augmentations
        self.num_strong_augmentations = num_strong_augmentations

        # flags
        self.train_label_guessing = train_label_guessing
        self.is_strong_augment_x = is_strong_augment_x

    @property
    def total_num_augmentations(self) -> int:
        return self.num_augmentations + self.num_strong_augmentations

    @torch.no_grad()
    def __call__(self,
                 x_augmented: Tensor,
                 x_strong_augmented: Optional[Tensor],
                 x_targets_one_hot: Tensor,
                 u_augmented: List[Tensor],
                 u_strong_augmented: List[Tensor],
                 u_true_targets_one_hot: Tensor,
                 model: Module,
                 *args,
                 **kwargs) -> Dict[str, Tensor]:
        if self.is_strong_augment_x:
            x_inputs = x_strong_augmented
        else:
            x_inputs = x_augmented
        u_inputs = u_augmented + u_strong_augmented

        # label guessing with weakly augmented data
        pseudo_label_dict = self.guess_label(u_inputs=u_augmented, model=model)

        return self.postprocess(x_augmented=x_inputs,
                                x_targets_one_hot=x_targets_one_hot,
                                u_augmented=u_inputs,
                                q_guess=pseudo_label_dict["q_guess"],
                                u_true_targets_one_hot=u_true_targets_one_hot)

    @torch.no_grad()
    def preprocess(self,
                   x_inputs: Tensor,
                   x_strong_inputs: Tensor,
                   x_targets: Tensor,
                   u_inputs: Tensor,
                   u_strong_inputs: Tensor,
                   u_true_targets: Tensor) -> Dict[str, Union[Optional[Tensor], List[Tensor]]]:
        # convert targets to one-hot
        x_targets_one_hot = F.one_hot(x_targets, num_classes=self.num_classes).type_as(x_inputs)
        u_true_targets_one_hot = F.one_hot(u_true_targets, num_classes=self.num_classes).type_as(x_inputs)

        # apply augmentations
        x_augmented = self.augmenter(x_inputs)
        u_augmented = [self.augmenter(u_inputs) for _ in range(self.num_augmentations)]

        if self.strong_augmenter is not None:
            x_strong_augmented = self.strong_augmenter(x_strong_inputs)
            u_strong_augmented = [self.strong_augmenter(u_strong_inputs) for _ in range(self.num_strong_augmentations)]
        else:
            x_strong_augmented = None
            u_strong_augmented = []

        return dict(x_augmented=x_augmented,
                    x_strong_augmented=x_strong_augmented,
                    x_targets_one_hot=x_targets_one_hot,
                    u_augmented=u_augmented,
                    u_strong_augmented=u_strong_augmented,
                    u_true_targets_one_hot=u_true_targets_one_hot)

    def guess_label(self, u_inputs: Sequence[Tensor], model: Module) -> Dict[str, Tensor]:
        # label guessing
        q_guess = label_guessing(batches=u_inputs, model=model, is_train_mode=self.train_label_guessing)
        q_guess = sharpen(q_guess, self.temperature)

        return dict(q_guess=q_guess)

    def postprocess(self,
                    x_augmented: Tensor,
                    x_targets_one_hot: Tensor,
                    u_augmented: List[Tensor],
                    q_guess: Tensor,
                    u_true_targets_one_hot: Tensor) -> Dict[str, Tensor]:
        # concat the unlabeled data and targets
        u_augmented = torch.cat(u_augmented, dim=0)
        q_guess = torch.cat([q_guess for _ in range(self.total_num_augmentations)], dim=0)
        q_true = torch.cat([u_true_targets_one_hot for _ in range(self.total_num_augmentations)], dim=0)
        assert len(u_augmented) == len(q_guess) == len(q_true)

        return self.mixup(x_augmented=x_augmented,
                          x_targets_one_hot=x_targets_one_hot,
                          u_augmented=u_augmented,
                          q_guess=q_guess,
                          q_true=q_true)

    def mixup(self,
              x_augmented: Tensor,
              x_targets_one_hot: Tensor,
              u_augmented: Tensor,
              q_guess: Tensor,
              q_true: Tensor) -> Dict[str, Tensor]:
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

        # the official version use the same lambda ~ sampled from Beta(alpha, alpha) for both
        # labeled and unlabeled inputs
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)

        wx_mixed = lam * wx + (1 - lam) * wx_shuffled
        wy_mixed = lam * wy + (1 - lam) * wy_shuffled
        wq_mixed = lam * wq + (1 - lam) * wq_shuffled

        x_mixed, p_mixed = wx_mixed[:len(x_augmented)], wy_mixed[:len(x_augmented)]
        u_mixed, q_mixed = wx_mixed[len(x_augmented):], wy_mixed[len(x_augmented):]
        q_true_mixed = wq_mixed[len(x_augmented):]

        return dict(x_mixed=x_mixed,
                    p_mixed=p_mixed,
                    u_mixed=u_mixed,
                    q_mixed=q_mixed,
                    q_true_mixed=q_true_mixed)
