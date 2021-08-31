from .pair_loss import PairLoss
from .utils import bha_coeff, bha_coeff_distance, l2_distance
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

from .types import SimilarityType, DistanceType, DistanceLossType


def get_similarity_metric(similarity_type: str) -> Tuple[SimilarityType, str]:
    """

    Args:
        similarity_type: the type of the similarity function

    Returns: similarity function, string indicating the type of the similarity (from [logit, prob, feature])

    """
    if similarity_type == "bhc":
        return bha_coeff, "prob"

    else:
        raise NotImplementedError(f"\"{similarity_type}\" is not a supported similarity type")


def get_distance_loss_metric(distance_loss_type: str) -> Tuple[DistanceLossType, str]:
    """


    Args:
        distance_loss_type: the type of the distance loss function

    Returns: distance loss function, string indicating the type of the loss (from [logit, prob])

    """
    if distance_loss_type == "bhc":
        return bha_coeff_loss, "logit"

    elif distance_loss_type == "l2":
        return l2_dist_loss, "prob"

    elif distance_loss_type == "entropy":
        return softmax_cross_entropy_loss, "logit"

    else:
        raise NotImplementedError(f"\"{distance_loss_type}\" is not a supported distance loss type")


def build_supervised_loss(args: Namespace) -> SupervisedLoss:
    return SupervisedLoss(reduction="mean")


def build_unsupervised_loss(args: Namespace) -> UnsupervisedLoss:
    return UnsupervisedLoss(
        loss_type=args.u_loss_type,
        loss_thresholded=args.u_loss_thresholded,
        confidence_threshold=args.confidence_threshold,
        reduction="mean")


def build_pair_loss(args: Namespace, reduction: str = "mean") -> PairLoss:
    similarity_metric, similarity_type = get_similarity_metric(args.similarity_type)
    distance_loss_metric, distance_loss_type = get_distance_loss_metric(args.distance_loss_type)

    return PairLoss(
        similarity_metric=similarity_metric,
        distance_loss_metric=distance_loss_metric,
        confidence_threshold=args.confidence_threshold,
        similarity_threshold=args.similarity_threshold,
        similarity_type=similarity_type,
        distance_loss_type=distance_loss_type,
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
