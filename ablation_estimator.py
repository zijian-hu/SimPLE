from torch.nn import functional as F

from simple_estimator import SimPLEEstimator

# for type hint
from typing import Tuple, Dict
from torch import Tensor
from loss.types import LossInfoType


class AblationEstimator(SimPLEEstimator):
    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Tuple[Tensor, LossInfoType]:
        outputs = self.preprocess_batch(batch, batch_idx)

        model_outputs = self.compute_train_logits(x_inputs=outputs["x_inputs"])

        outputs.update(model_outputs)

        # calculate loss
        return self.compute_train_loss(
            x_logits=outputs["x_logits"],
            x_targets=outputs["x_targets"]
        )

    def preprocess_batch(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Dict[str, Tensor]:
        # unpack batch
        (x_inputs, x_targets), (_, _) = batch

        # load data to device
        x_inputs = x_inputs.to(self.device)
        x_targets = x_targets.to(self.device)

        # apply augmentations
        x_inputs = self.augmenter(x_inputs)

        return dict(
            x_inputs=x_inputs,
            x_targets=x_targets,
        )

    def compute_train_logits(self, x_inputs: Tensor, *args: Tensor) -> Dict[str, Tensor]:
        return dict(x_logits=self.model(x_inputs))

    def compute_train_loss(self,
                           x_logits: Tensor,
                           x_targets: Tensor,
                           *args: Tensor,
                           **kwargs) -> Tuple[Tensor, LossInfoType]:
        loss = F.cross_entropy(x_logits, x_targets, reduction="mean")

        log_info = {
            "loss": loss.detach().clone(),
            "loss_x": loss.detach().clone(),
        }

        return loss, {"log": log_info, "plot": {}}
