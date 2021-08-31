import wandb

import sys

from .logger import Logger

# for type hint
from typing import Any, Dict, Optional, Union, Sequence
from argparse import Namespace

from torch.nn import Module


class WandbLogger(Logger):
    def __init__(self,
                 log_dir: str,
                 config: Union[Namespace, Dict[str, Any]],
                 name: str,
                 tags: Sequence[str],
                 notes: str,
                 entity: str,
                 project: str,
                 mode: str = "offline",
                 resume: Union[bool, str] = False):
        super().__init__(
            log_dir=log_dir,
            config=config,
            log_info_key_map={
                "top1_acc": "mean_acc",
                "top5_acc": "mean_top5_acc",
            })

        self.name = name
        self.config = config
        self.tags = tags
        self.notes = notes
        self.entity = entity
        self.project = project
        self.mode = mode
        self.resume = resume

        self.is_init = False

    def _init_wandb(self):
        if not self.is_init:
            wandb.init(
                name=self.name,
                config=self.config,
                project=self.project,
                entity=self.entity,
                dir=self.log_dir,
                tags=self.tags,
                notes=self.notes,
                mode=self.mode,
                resume=self.resume)
            # update is_init flag
            self.is_init = True

            # update config if resumed
            if bool(self.resume):
                self.log(self.config, is_config=True)

    def log(self,
            log_info: Dict[str, Any],
            step: Optional[int] = None,
            is_summary: bool = False,
            is_config: bool = False,
            is_commit: Optional[bool] = None,
            prefix: Optional[str] = None,
            log_info_override: Optional[Dict[str, Any]] = None,
            **kwargs):
        if not self.is_init:
            self._init_wandb()

        # process log_info
        log_info = self.process_log_info(log_info, prefix=prefix, log_info_override=log_info_override)

        if len(log_info) == 0:
            return

        if is_summary:
            # for log_info_key, log_info_value in log_info.items():
            #     wandb.run.summary[log_info_key] = log_info_value
            wandb.run.summary.update(log_info)
        elif is_config:
            wandb.run.config.update(log_info, allow_val_change=True)
        else:
            log_info, plot_info = self.separate_plot(log_info)
            wandb.log(plot_info, commit=False, step=step)
            wandb.log(log_info, commit=is_commit, step=step)

        # invoke all log hook functions
        self.call_log_hooks(log_info)

    def watch(self, model: Module, **kwargs):
        if not self.is_init:
            self._init_wandb()

        wandb.watch(model, **kwargs)

    def save(self, output_path: str):
        if not self.is_init:
            self._init_wandb()

        if sys.platform != "win32":
            # TODO: remove the if condition once found a solution
            # Windows requires elevated access to
            wandb.save(output_path)
