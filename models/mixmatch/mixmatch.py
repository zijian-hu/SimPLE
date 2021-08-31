from .mixmatch_base import MixMatchBase

# for type hint
from torch.nn import Module


class MixMatch(MixMatchBase):
    def __init__(self,
                 augmenter: Module,
                 num_classes: int,
                 temperature: float,
                 num_augmentations: int,
                 alpha: float,
                 train_label_guessing: bool):
        super(MixMatch, self).__init__(augmenter=augmenter,
                                       strong_augmenter=None,
                                       num_classes=num_classes,
                                       temperature=temperature,
                                       num_augmentations=num_augmentations,
                                       num_strong_augmentations=0,
                                       alpha=alpha,
                                       is_strong_augment_x=False,
                                       train_label_guessing=train_label_guessing)
