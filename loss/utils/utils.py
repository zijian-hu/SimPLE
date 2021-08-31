import torch

# for type hint
from typing import Union, Optional, Sequence, Callable
from torch import Tensor

ScalarType = Union[int, float, bool]


def reduce_tensor(inputs: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return torch.mean(inputs)

    elif reduction == 'sum':
        return torch.sum(inputs)

    return inputs


def to_tensor(data: Union[ScalarType, Sequence[ScalarType]],
              dtype: Optional[torch.dtype] = None,
              device: Optional[Union[torch.device, str]] = None,
              tensor_like: Optional[Tensor] = None) -> Tensor:
    if tensor_like is not None:
        dtype = tensor_like.dtype if dtype is None else dtype
        device = tensor_like.device if device is None else device

    return torch.tensor(data, dtype=dtype, device=device)


def bha_coeff_log_prob(log_p: Tensor, log_q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of log(p) and log(q); the more similar the larger the coefficient
    :param log_p: (batch_size, num_classes) first log prob distribution
    :param log_q: (batch_size, num_classes) second log prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    # numerical unstable version
    # coefficient = torch.sum(torch.sqrt(p * q), dim=dim)
    # numerical stable version
    coefficient = torch.sum(torch.exp((log_p + log_q) / 2), dim=dim)

    return reduce_tensor(coefficient, reduction)


def bha_coeff(p: Tensor, q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param p: (batch_size, num_classes) first prob distribution
    :param q: (batch_size, num_classes) second prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_p = torch.log(p)
    log_q = torch.log(q)

    return bha_coeff_log_prob(log_p, log_q, dim=dim, reduction=reduction)


def bha_coeff_distance(p: Tensor, q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param p: (batch_size, num_classes) model predictions of the data
    :param q: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    return 1. - bha_coeff(p, q, dim=dim, reduction=reduction)


def l2_distance(x: Tensor, y: Tensor, dim: int, **kwargs) -> Tensor:
    return torch.norm(x - y, p=2, dim=dim)


def pairwise_apply(p: Tensor, q: Tensor, func: Callable, *args, **kwargs) -> Tensor:
    """

    Args:
        p: (batch_size, num_classes) first prob distribution
        q: (batch_size, num_classes) second prob distribution
        func: function to be applied on p and q

    Returns: a matrix of pair-wise result between each element of p and q

    """
    p = p.unsqueeze(-1)
    q = q.T.unsqueeze(0)
    return func(p, q, *args, **kwargs)
