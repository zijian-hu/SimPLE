import torch
from torch.nn import functional as F

from .utils import reduce_tensor, bha_coeff_log_prob, l2_distance

# for type hint
from torch import Tensor


def softmax_cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = 'mean') -> Tensor:
    """
    :param logits: (labeled_batch_size, num_classes) model output of the labeled data
    :param targets: (labeled_batch_size, num_classes) labels distribution for the data
    :param dim: the dimension or dimensions to reduce
    :param reduction: choose from 'mean', 'sum', and 'none'
    :return:
    """
    loss = -torch.sum(F.log_softmax(logits, dim=dim) * targets, dim=dim)

    return reduce_tensor(loss, reduction)


def mse_loss(prob: Tensor, targets: Tensor, reduction: str = 'mean', **kwargs) -> Tensor:
    return F.mse_loss(prob, targets, reduction=reduction)


def bha_coeff_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param logits: (batch_size, num_classes) model predictions of the data
    :param targets: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_probs = F.log_softmax(logits, dim=dim)
    log_targets = torch.log(targets)

    # since BC(P,Q) is maximized when P and Q are the same, we minimize 1 - B(P,Q)
    return 1. - bha_coeff_log_prob(log_probs, log_targets, dim=dim, reduction=reduction)


def l2_dist_loss(probs: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    loss = l2_distance(probs, targets, dim=dim)

    return reduce_tensor(loss, reduction)


class SupervisedLoss:
    def __init__(self, reduction: str = 'mean'):
        self.loss_use_prob = False
        self.loss_fn = softmax_cross_entropy_loss

        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        loss_input = probs if self.loss_use_prob else logits
        loss = self.loss_fn(loss_input, targets, dim=1, reduction=self.reduction)

        return loss


class UnsupervisedLoss:
    def __init__(self,
                 loss_type: str,
                 loss_thresholded: bool = False,
                 confidence_threshold: float = 0.,
                 reduction: str = "mean"):
        if loss_type in ["entropy", "cross entropy"]:
            self.loss_use_prob = False
            self.loss_fn = softmax_cross_entropy_loss
        else:
            self.loss_use_prob = True
            self.loss_fn = mse_loss

        self.loss_thresholded = loss_thresholded
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        loss_input = probs if self.loss_use_prob else logits
        loss = self.loss_fn(loss_input, targets, dim=1, reduction="none")

        if self.loss_thresholded:
            targets_mask = (targets.max(dim=1).values > self.confidence_threshold)

            if len(loss.shape) > 1:
                # mse_loss returns a matrix, need to reshape mask
                targets_mask = targets_mask.view(-1, 1)

            loss *= targets_mask.float()

        return reduce_tensor(loss, reduction=self.reduction)
