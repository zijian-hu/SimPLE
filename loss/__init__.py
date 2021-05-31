from .pair_loss import PairLoss
from .utils import (bha_coeff, bha_coeff_distance, hel_dist, l2_distance)
from .loss import (SupervisedLoss, UnsupervisedLoss, softmax_cross_entropy_loss, bha_coeff_loss,
                   l2_dist_loss)

# modules
from . import utils
from . import types
from . import pair_loss
from . import visualization

# for type hint
from typing import Tuple
from argparse import Namespace

from .types import LossOutType, SimilarityType, DistanceType, DistanceLossType


def get_similarity_metric(similarity_type: str) -> SimilarityType:
    # other similarity functions can be added here
    return bha_coeff


def get_distance_loss_metric(distance_loss_type: str) -> Tuple[DistanceLossType, bool]:
    # other distance loss functions can be added here
    if distance_loss_type == "l2":
        distance_use_prob = True
        distance_loss_metric = l2_dist_loss

    else:
        distance_use_prob = False
        distance_loss_metric = bha_coeff_loss

    return distance_loss_metric, distance_use_prob


def build_supervised_loss(args: Namespace) -> SupervisedLoss:
    # other loss functions can be added here
    return SupervisedLoss(reduction="mean")


def build_unsupervised_loss(args: Namespace) -> UnsupervisedLoss:
    return UnsupervisedLoss(
        loss_type=args.u_loss_type,
        loss_thresholded=args.u_loss_thresholded,
        confidence_threshold=args.confidence_threshold,
        reduction="mean")


def build_pair_loss(args: Namespace, reduction: str = "mean") -> PairLoss:
    similarity_metric = get_similarity_metric(args.similarity_type)
    distance_loss_metric, distance_use_prob = get_distance_loss_metric(args.distance_loss_type)

    return PairLoss(
        similarity_metric=similarity_metric,
        distance_loss_metric=distance_loss_metric,
        confidence_threshold=args.confidence_threshold,
        similarity_threshold=args.similarity_threshold,
        distance_use_prob=distance_use_prob,
        reduction=reduction)


__all__ = [
    # modules
    "utils",
    "types",
    "pair_loss",
    "visualization",

    # classes
    "SupervisedLoss",
    "UnsupervisedLoss",

    # functions
    "get_similarity_metric",
    "get_distance_loss_metric",

    # loss functions
    "build_supervised_loss",
    "build_unsupervised_loss",
    "build_pair_loss",
]
