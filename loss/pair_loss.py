import torch

from .utils import get_pair_indices

# for type hint
from typing import Optional
from torch import Tensor

from .types import LossOutType, SimilarityType, DistanceLossType


class PairLoss:
    def __init__(self,
                 similarity_metric: SimilarityType,
                 distance_loss_metric: DistanceLossType,
                 confidence_threshold: float = 0.,
                 similarity_threshold: float = 0.9,
                 distance_use_prob: bool = True,
                 reduction: str = "mean"):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.distance_use_prob = distance_use_prob

        self.reduction = reduction

        self.get_similarity = similarity_metric
        self.get_distance_loss = distance_loss_metric

    def __call__(self,
                 logits: Tensor,
                 probs: Tensor,
                 targets: Tensor,
                 true_targets: Tensor,
                 indices: Optional[Tensor] = None) -> LossOutType:
        """

        Args:
            logits: (batch_size, num_classes) predictions of batch data
            probs: (batch_size, num_classes) softmax probs logits
            targets: (batch_size, num_classes) one hot labels
            true_targets: (batch_size, num_classes) one hot ground truth labels; used for visualization only

        Returns: None if no pair satisfy the constraints

        """
        if indices is None:
            indices = get_pair_indices(targets, ordered_pair=True)
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]

        logits_j = logits[j_indices]
        probs_j = probs[j_indices]
        targets_i = targets[i_indices]
        targets_j = targets[j_indices]

        targets_max_prob = targets.max(dim=1)[0]
        targets_i_max_prob = targets_max_prob[i_indices]

        conf_mask = targets_i_max_prob > self.confidence_threshold

        sim: Tensor = self.get_similarity(targets_i, targets_j, dim=1)

        factor = conf_mask.float() * torch.threshold(sim, self.similarity_threshold, 0)

        if self.distance_use_prob:
            loss_input = probs_j
        else:
            loss_input = logits_j
        distance_ij = self.get_distance_loss(loss_input, targets_i, dim=1, reduction='none')

        loss = factor * distance_ij

        if self.reduction == "mean":
            loss = torch.sum(loss) / total_size
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss, {"log": {}, "plot": {}}
