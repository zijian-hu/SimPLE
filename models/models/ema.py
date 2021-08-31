import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
import warnings
from contextlib import contextmanager

from .utils import unwrap_model

# for type hint
from torch import Tensor
from typing import Dict


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        # adapted from https://fyubang.com/2019/06/01/ema/
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            warnings.warn("EMA update should only be called during training")
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor, return_feature: bool = False) -> Tensor:
        if self.training:
            return self.model(inputs, return_feature)
        else:
            return self.shadow(inputs, return_feature)

    @contextmanager
    def data_parallel_switch(self):
        model = self.model
        shadow = self.shadow

        self.model = unwrap_model(self.model)
        self.shadow = unwrap_model(self.shadow)

        try:
            yield
        finally:
            self.model = model
            self.shadow = shadow

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        with self.data_parallel_switch():
            return super().state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict=True):
        with self.data_parallel_switch():
            super().load_state_dict(state_dict, strict)


class FullEMA(EMA):
    excluded_param_suffix = ["num_batches_tracked"]

    def __init__(self, model: nn.Module, decay: float):
        super().__init__(model, decay)

        self._excluded_param_names = set()

    @torch.no_grad()
    def update(self):
        if not self.training:
            warnings.warn("EMA update should only be called during training")
            return

        model_state_dict = self.model.state_dict()
        shadow_state_dict = self.shadow.state_dict()

        # check if both model contains the same set of keys
        assert model_state_dict.keys() == shadow_state_dict.keys()

        for name, param in model_state_dict.items():
            if name not in self.excluded_param_names:
                shadow_state_dict[name].sub_((1. - self.decay) * (shadow_state_dict[name] - param))
            else:
                shadow_state_dict[name].copy_(param)

    @staticmethod
    def is_ema_exclude(param_name: str) -> bool:
        return any([param_name.endswith(suffix) for suffix in FullEMA.excluded_param_suffix])

    @property
    def excluded_param_names(self):
        if len(self._excluded_param_names) == 0:
            for name, param in self.model.state_dict().items():
                if self.is_ema_exclude(name):
                    self._excluded_param_names.add(name)

        return self._excluded_param_names
