import torch
import numpy as np
from kornia.utils import confusion_matrix
import plotly.figure_factory as ff
import wandb

from .utils import to_tensor, get_pair_indices

# for type hint
from typing import Tuple, Union, Dict
from torch import Tensor
from plotly.graph_objects import Figure

from .types import SimilarityType, LogDictType, PlotDictType, LossInfoType


@torch.no_grad()
def get_pair_info(targets: Tensor,
                  true_targets: Tensor,
                  similarity_metric: SimilarityType,
                  confidence_threshold: float,
                  similarity_threshold: float,
                  return_plot_info: bool = False) -> LossInfoType:
    indices = get_pair_indices(targets, ordered_pair=True)
    i_indices, j_indices = indices[:, 0], indices[:, 1]
    targets_i, targets_j = targets[i_indices], targets[j_indices]

    targets_max_prob = targets.max(dim=1)[0]

    conf_mask = targets_max_prob[i_indices] > confidence_threshold
    sim: Tensor = similarity_metric(targets_i, targets_j, dim=1)

    sim_mask = (sim > similarity_threshold)
    final_mask = (conf_mask & sim_mask)
    total_size = len(indices)

    log_info, plot_info = get_loss_info(
        all_high_conf_mask=conf_mask,
        all_high_sim_mask=sim_mask,
        final_mask=final_mask,
        true_targets=true_targets,
        indices=indices,
        total_size=total_size,
        return_plot_info=return_plot_info)

    # log extra metrics
    extra_log_info, extra_plot_info = get_pair_extra_info(
        targets_max_prob=targets_max_prob,
        i_indices=i_indices,
        j_indices=j_indices,
        similarities=sim,
        final_mask=final_mask)

    log_info.update(extra_log_info)
    plot_info.update(extra_plot_info)

    return {"log": log_info, "plot": plot_info}


@torch.no_grad()
def get_loss_info(all_high_conf_mask: Tensor,
                  all_high_sim_mask: Tensor,
                  final_mask: Tensor,
                  true_targets: Tensor,
                  total_size: int,
                  indices: Tensor,
                  return_plot_info: bool = False,
                  **kwargs) -> Tuple[LogDictType, PlotDictType]:
    # prepare log info
    total_high_conf = all_high_conf_mask.sum()
    total_high_sim = all_high_sim_mask.sum()
    total_thresholded = final_mask.sum()

    thresholded_indices = indices[final_mask]
    true_labels = true_targets.max(dim=1)[0]
    true_labels_i = true_labels[thresholded_indices[:, 0]]
    true_labels_j = true_labels[thresholded_indices[:, 1]]

    total_true_pairs = (true_labels_i == true_labels_j).sum().float()
    if len(thresholded_indices) == 0:
        true_pair_given_thresholded = to_tensor(0., tensor_like=total_true_pairs)
    else:
        true_pair_given_thresholded = (total_true_pairs / len(thresholded_indices))

    matrix = get_confusion_matrix(true_targets, final_mask, indices)
    normalized_matrix = matrix / matrix.sum()

    log_info = {
        "pair_loss/high_conf_ratio": (total_high_conf.float() / total_size),
        "pair_loss/high_sim_ratio": (total_high_sim.float() / total_size),
        "pair_loss/thresholded_ratio": (total_thresholded.float() / total_size),
        "pair_loss/true_pair_given_thresholded": true_pair_given_thresholded,

        # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # In binary classification, the count of true negatives is C[0,0], false negatives is C[1,0],
        # true positives is C[1,1] and false positives is C[0,1].
        "pair_loss/true_negative_pair_ratio": normalized_matrix[0, 0],
        "pair_loss/false_positives_pair_ratio": normalized_matrix[0, 1],
        "pair_loss/false_negatives_pair_ratio": normalized_matrix[1, 0],
        "pair_loss/true_positives_pair_ratio": normalized_matrix[1, 1],
    }

    plot_info = {}
    if return_plot_info:
        plot_info["pair_loss/pair_confusion_matrix"] = visualize(matrix.detach().cpu().numpy(),
                                                                 normalized_matrix.detach().cpu().numpy())

    return log_info, plot_info


def get_confusion_matrix(true_targets: Tensor, y_pred: Tensor, indices: Tensor) -> Tensor:
    # TODO: support distributed training
    true_labels = true_targets.max(dim=1)[0]
    y_true = (true_labels[indices[:, 0]] == true_labels[indices[:, 1]])

    matrix = confusion_matrix(y_pred.view(1, -1), y_true.view(1, -1), num_classes=2, normalized=False)
    matrix.squeeze_()

    return matrix


def visualize(matrix: np.ndarray, normalized_matrix: np.ndarray, **kwargs) -> Figure:
    # see https://matplotlib.org/faq/usage_faq.html#coding-styles
    # see https://plotly.com/python/annotated-heatmap/
    # see https://plotly.com/python/builtin-colorscales/
    annotation_text = [
        [f"True Negative: {normalized_matrix[0, 0]:.3%}", f"False Positive: {normalized_matrix[0, 1]:.3%}"],
        [f"False Negative: {normalized_matrix[1, 0]:.3%}", f"True Positive: {normalized_matrix[1, 1]:.3%}"]
    ]
    x_axis_text = ["Predicted False", "Predicted True"]
    y_axis_text = ["False", "True"]
    fig = ff.create_annotated_heatmap(matrix, x=x_axis_text, y=y_axis_text, annotation_text=annotation_text,
                                      colorscale='Plotly3')
    fig.update_layout({
        'xaxis': {'title': {'text': "Predicted"}},
        'yaxis': {'title': {'text': "Ground Truth"}},
    })

    return fig


@torch.no_grad()
def get_pair_extra_info(targets_max_prob: Tensor,
                        i_indices: Tensor,
                        j_indices: Tensor,
                        similarities: Tensor,
                        final_mask: Tensor) -> Tuple[LogDictType, PlotDictType]:
    def mean_std_max_min(t: Union[Tensor, np.ndarray], prefix: str = "") -> Dict[str, Union[Tensor, np.ndarray]]:
        return {
            f"{prefix}/mean": t.mean() if t.numel() > 0 else to_tensor(0, tensor_like=t),
            f"{prefix}/std": t.std() if t.numel() > 0 else to_tensor(0, tensor_like=t),
            f"{prefix}/max": t.max() if t.numel() > 0 else to_tensor(0, tensor_like=t),
            f"{prefix}/min": t.min() if t.numel() > 0 else to_tensor(0, tensor_like=t),
        }

    def detorch(t: Tensor) -> np.ndarray:
        return t.detach().cpu().clone().numpy()

    targets_i_max_prob = targets_max_prob[i_indices]
    targets_j_max_prob = targets_max_prob[j_indices]

    selected_sim = similarities[final_mask]
    selected_i_conf = targets_i_max_prob[final_mask]
    selected_j_conf = targets_j_max_prob[final_mask]

    selected_i_conf_stat = mean_std_max_min(selected_i_conf, prefix="selected_i_conf")
    selected_j_conf_stat = mean_std_max_min(selected_j_conf, prefix="selected_j_conf")
    selected_sim_stat = mean_std_max_min(selected_sim, prefix="selected_sim")

    selected_i_conf_hist = wandb.Histogram(detorch(selected_i_conf))
    selected_j_conf_hist = wandb.Histogram(detorch(selected_j_conf))
    selected_sim_hist = wandb.Histogram(detorch(selected_sim))

    log_info = {
        **selected_i_conf_stat,
        **selected_j_conf_stat,
        **selected_sim_stat,
    }
    plot_info = {
        "selected_i_conf_hist": selected_i_conf_hist,
        "selected_j_conf_hist": selected_j_conf_hist,
        "selected_sim_hist": selected_sim_hist,
    }

    return {f"pair_loss/{k}": v for k, v in log_info.items()}, \
           {f"pair_loss/{k}": v for k, v in plot_info.items()}
