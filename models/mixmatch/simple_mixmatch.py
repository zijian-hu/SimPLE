import torch

from .mixmatch_base import MixMatchBase

# for type hint
from typing import List, Dict
from torch import Tensor
from torch.nn import Module


class SimPLE(MixMatchBase):
    def __init__(self,
                 augmenter: Module,
                 strong_augmenter: Module,
                 num_classes: int,
                 temperature: float,
                 num_augmentations: int,
                 num_strong_augmentations: int,
                 is_strong_augment_x: bool,
                 train_label_guessing: bool):
        super(SimPLE, self).__init__(augmenter=augmenter,
                                     strong_augmenter=strong_augmenter,
                                     num_classes=num_classes,
                                     temperature=temperature,
                                     num_augmentations=num_augmentations,
                                     num_strong_augmentations=num_strong_augmentations,
                                     alpha=0.,
                                     is_strong_augment_x=is_strong_augment_x,
                                     train_label_guessing=train_label_guessing)

    @property
    def total_num_augmentations(self) -> int:
        return self.num_strong_augmentations

    @torch.no_grad()
    def __call__(self,
                 x_augmented: Tensor,
                 x_strong_augmented: Tensor,
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
        u_inputs = u_strong_augmented

        # label guessing with weakly augmented data
        pseudo_label_dict = self.guess_label(u_inputs=u_augmented, model=model)

        return self.postprocess(x_augmented=x_inputs,
                                x_targets_one_hot=x_targets_one_hot,
                                u_augmented=u_inputs,
                                q_guess=pseudo_label_dict["q_guess"],
                                u_true_targets_one_hot=u_true_targets_one_hot)

    def mixup(self,
              x_augmented: Tensor,
              x_targets_one_hot: Tensor,
              u_augmented: Tensor,
              q_guess: Tensor,
              q_true: Tensor) -> Dict[str, Tensor]:
        # SimPLE do not use mixup
        return dict(x_mixed=x_augmented,
                    p_mixed=x_targets_one_hot,
                    u_mixed=u_augmented,
                    q_mixed=q_guess,
                    q_true_mixed=q_true)
