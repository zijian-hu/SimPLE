import torch
from torch import onnx

from .models.utils import *

# for type hint
from typing import Optional, Sequence, List, Generator, Any, Union, Set, Tuple, Dict
from torch import Tensor
from torch.nn import Module, Parameter


@torch.no_grad()
def get_accuracy(output: Tensor, target: Tensor, top_k: Sequence[int] = (1,)) -> List[Tensor]:
    # see https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = (correct[:, :k].sum(dim=1, keepdim=False) > 0).float()
        res.append(correct_k.sum() / batch_size)
    return res


def interleave_offsets(batch_size: int, num_unlabeled: int) -> List[int]:
    # TODO: scrutiny
    groups = [batch_size // (num_unlabeled + 1)] * (num_unlabeled + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def interleave(xy: Sequence[Tensor], batch_size: int) -> List[Tensor]:
    # TODO: scrutiny
    num_unlabeled = len(xy) - 1
    offsets = interleave_offsets(batch_size, num_unlabeled)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(num_unlabeled + 1)] for v in xy]
    for i in range(1, num_unlabeled + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def get_gradient_norm(model: Module, grad_enabled: bool = False) -> Tensor:
    with torch.set_grad_enabled(grad_enabled):
        return sum([(p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None])


def get_weight_norm(model: Module, grad_enabled: bool = False) -> Tensor:
    with torch.set_grad_enabled(grad_enabled):
        return sum([(p.detach() ** 2).sum() for p in model.parameters() if p.data is not None])


def split_classifier_params(model: Module, classifier_prefix: Union[str, Set[str]]) \
        -> Tuple[List[Parameter], List[Parameter]]:
    if not isinstance(classifier_prefix, Set):
        classifier_prefix = {classifier_prefix}

    # build tuple for multiple prefix matching
    classifier_prefix = tuple(sorted(f"{prefix}." for prefix in classifier_prefix))

    embedder_weights = []
    classifier_weights = []

    for k, v in model.named_parameters():
        if k.startswith(classifier_prefix):
            classifier_weights.append(v)
        else:
            embedder_weights.append(v)

    return embedder_weights, classifier_weights


def set_model_mode(model: Module, mode: Optional[bool]) -> Generator[Any, Any, None]:
    """
    A context manager to temporarily set the training mode of ‘model’ to ‘mode’, resetting it when we exit the
    with-block. A no-op if mode is None

    Args:
        model: the model
        mode: a bool or None

    Returns:

    """
    if hasattr(onnx, "select_model_mode_for_export"):
        # In PyTorch 1.6+, set_training is changed to select_model_mode_for_export
        return onnx.select_model_mode_for_export(model, mode)
    else:
        return onnx.set_training(model, mode)


def consume_prefix_in_state_dict_if_present(state_dict: Dict[str, Any], prefix: str):
    r"""copied from https://github.com/pytorch/pytorch/blob/255494c2aa1fcee7e605a6905be72e5b8ccf4646/torch/nn/modules/utils.py#L37-L67

    Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


__all__ = [
    "get_accuracy",
    "interleave",
    "get_gradient_norm",
    "get_weight_norm",
    "split_classifier_params",
    "set_model_mode",
    "unwrap_model",
    "consume_prefix_in_state_dict_if_present",
]
