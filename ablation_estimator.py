from torch.nn import functional as F

from simple_estimator import SimPLEEstimator

# for type hint
from typing import Tuple
from torch import Tensor
from loss.types import LossOutType


class AblationEstimator(SimPLEEstimator):
    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> LossOutType:
        x_inputs, x_targets = self.preprocess_batch(batch, batch_idx)

        x_logits = self.compute_train_logits(x_inputs)

        # calculate loss
        return self.compute_train_loss(x_logits, x_targets)

    def preprocess_batch(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Tuple[Tensor, ...]:
        # unpack batch
        (x_inputs, x_targets), (_, _) = batch

        # load data to device
        x_inputs = x_inputs.to(self.device)
        x_targets = x_targets.to(self.device)

        # apply augmentations
        x_inputs = self.augmenter(x_inputs)

        return x_inputs, x_targets

    def compute_train_logits(self, x_inputs: Tensor, *args: Tensor) -> Tensor:
        return self.model(x_inputs)

    def compute_train_loss(self,
                           x_logits: Tensor,
                           x_targets: Tensor,
                           *args: Tensor,
                           return_plot_info: bool = False) -> LossOutType:
        loss = F.cross_entropy(x_logits, x_targets, reduction="mean")

        log_info = {
            "loss": loss,
            "loss_x": loss,
        }

        return loss, {"log": log_info, "plot": {}}
