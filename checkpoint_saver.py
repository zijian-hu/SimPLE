import torch
import numpy as np

from pathlib import Path
import re
import warnings

from utils import find_checkpoint_path, find_all_files
from utils.metrics import MetricMode, MetricMonitor

# for type hint
from typing import Dict, Any, Optional, Union
from simple_estimator import SimPLEEstimator
from utils import Logger


class CheckpointSaver:
    def __init__(self,
                 estimator: SimPLEEstimator,
                 logger: Logger,
                 checkpoint_metric: str,
                 best_checkpoint_str: str,
                 best_checkpoint_pattern: str,
                 latest_checkpoint_str: str,
                 latest_checkpoint_pattern: str,
                 delayed_best_model_saving: bool = True):
        """

        Args:
            estimator: Estimator, used to get experiment related data
            logger: Logger, used for logging
            checkpoint_metric: save model when the best value of this key has changed.
            best_checkpoint_str: path str format to save best checkpoint file
            best_checkpoint_pattern: regex pattern used to find the best checkpoint file
            latest_checkpoint_str: path str format to save best checkpoint file
            latest_checkpoint_pattern: regex pattern used to find the latest checkpoint file
            delayed_best_model_saving: if True, save best model after calling save_latest_checkpoint()
        """
        self.absolute_best_path = "best_checkpoint.pth"

        # metrics to keep track of
        self.monitor = MetricMonitor()
        self.monitor.track(key="mean_acc",
                           best_value=-np.inf,
                           mode=MetricMode.MAX,
                           prefix="test")
        self.monitor.track(key="mean_acc",
                           best_value=-np.inf,
                           mode=MetricMode.MAX,
                           prefix="validation")

        self.checkpoint_metric = checkpoint_metric

        # checkpoint path patterns
        self.best_checkpoint_str = best_checkpoint_str
        self.best_checkpoint_pattern = re.compile(best_checkpoint_pattern)

        self.latest_checkpoint_str = latest_checkpoint_str
        self.latest_checkpoint_pattern = re.compile(latest_checkpoint_pattern)

        # save estimator and logger
        # this will recover best metrics and register log hooks
        self.estimator = estimator
        self.logger = logger

        # assign flags
        self.delayed_save_best_model = delayed_best_model_saving
        self.is_best_model = False

    @property
    def estimator(self) -> SimPLEEstimator:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: SimPLEEstimator) -> None:
        self._estimator = estimator

        # recover best value
        checkpoint_path = self.estimator.exp_args.checkpoint_path
        if checkpoint_path is not None:
            print(f"Recovering best metrics from {checkpoint_path}...")
            self.recover_metrics(torch.load(checkpoint_path, map_location=self.device))

    @property
    def logger(self) -> Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        self._logger = logger

        # register log hooks
        print("Registering log hooks...")
        self.logger.register_log_hook(self.update_best_metric, logger=self.logger)

    @property
    def checkpoint_metric(self) -> str:
        return self._checkpoint_metric

    @checkpoint_metric.setter
    def checkpoint_metric(self, checkpoint_metric: str) -> None:
        assert checkpoint_metric in self.monitor, f"{checkpoint_metric} is not in metric monitor"

        self._checkpoint_metric = checkpoint_metric

    @property
    def log_dir(self) -> str:
        return self.estimator.exp_args.log_dir

    @property
    def best_full_checkpoint_str(self) -> str:
        return str(Path(self.log_dir) / self.best_checkpoint_str)

    @property
    def latest_full_checkpoint_str(self) -> str:
        return str(Path(self.log_dir) / self.latest_checkpoint_str)

    @property
    def device(self) -> torch.device:
        return self.estimator.device

    @property
    def global_step(self) -> int:
        return self.estimator.global_step

    @property
    def num_latest_checkpoints_kept(self) -> Optional[int]:
        return self.estimator.exp_args.num_latest_checkpoints_kept

    @property
    def is_save_latest_checkpoint(self) -> bool:
        return self.num_latest_checkpoints_kept is None or self.num_latest_checkpoints_kept > 0

    @property
    def is_remove_old_checkpoint(self) -> bool:
        return self.num_latest_checkpoints_kept is not None and self.num_latest_checkpoints_kept > 0

    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        checkpoint_path: Union[str, Path],
                        is_logger_save: bool = False) -> Path:
        checkpoint_path = str(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint saved to \"{checkpoint_path}\"", flush=True)

        if is_logger_save:
            self.logger.save(checkpoint_path)

        return Path(checkpoint_path)

    def save_best_checkpoint(self,
                             checkpoint: Optional[Dict[str, any]] = None,
                             is_logger_save: bool = False,
                             **kwargs) -> Path:
        if checkpoint is None:
            checkpoint = self.get_checkpoint()

        checkpoint_path = self.save_checkpoint(checkpoint_path=self.best_full_checkpoint_str.format(**kwargs),
                                               checkpoint=checkpoint,
                                               is_logger_save=is_logger_save)
        # reset flag
        self.is_best_model = False

        return checkpoint_path

    def save_latest_checkpoint(self,
                               checkpoint: Optional[Dict[str, any]] = None,
                               is_logger_save: bool = False,
                               **kwargs) -> Optional[Path]:
        checkpoint_path: Optional[Path] = None

        if self.is_save_latest_checkpoint:
            if checkpoint is None:
                checkpoint = self.get_checkpoint()

            # save new checkpoint
            checkpoint_path = self.save_checkpoint(checkpoint_path=self.latest_full_checkpoint_str.format(**kwargs),
                                                   checkpoint=checkpoint,
                                                   is_logger_save=is_logger_save)

            # cleanup old checkpoints
            self.cleanup_checkpoints()

        if self.delayed_save_best_model and self.is_best_model:
            self.save_best_checkpoint(**kwargs)

        return checkpoint_path

    def get_checkpoint(self) -> Dict[str, Any]:
        checkpoint = self.estimator.get_checkpoint()

        # add best metrics
        checkpoint.update({"monitor_state": self.monitor.state_dict()})

        return checkpoint

    def update_best_checkpoint(self) -> None:
        """
        Update the logged metrics for the best checkpoint

        Returns:

        """
        best_checkpoint_path = self.find_best_checkpoint_path()

        if best_checkpoint_path is None:
            warnings.warn("Cannot find best checkpoint")
            return

        best_checkpoint_path = str(best_checkpoint_path)
        best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device)

        # update best metrics
        best_checkpoint.update({"monitor_state": self.monitor.state_dict()})

        self.save_checkpoint(checkpoint_path=str(Path(self.log_dir) / self.absolute_best_path),
                             checkpoint=best_checkpoint)

    def find_best_checkpoint_path(self, checkpoint_dir: Optional[str] = None, ignore_absolute_best: bool = True) \
            -> Optional[Path]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        abs_best_path = Path(checkpoint_dir) / self.absolute_best_path

        if not ignore_absolute_best and abs_best_path.is_file():
            # if not ignoring absolute best path and the path is a file, return the absolute best file path
            return abs_best_path

        checkpoint_path = find_checkpoint_path(checkpoint_dir, step_filter=self.best_checkpoint_pattern)

        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint_path(checkpoint_dir=checkpoint_dir)

        return checkpoint_path

    def find_latest_checkpoint_path(self, checkpoint_dir: Optional[str] = None) -> Optional[Path]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        return find_checkpoint_path(checkpoint_dir, step_filter=self.latest_checkpoint_pattern)

    def update_best_metric(self, log_info: Dict[str, Any], logger: Logger) -> None:
        updated_dict = self.monitor.update_metrics(log_info)

        for updated_key, new_best_value in updated_dict.items():
            metric_dict = self.monitor[updated_key]

            translated_key = metric_dict["key"]

            # if new_best_value is better than current best value
            logger.log({translated_key: new_best_value}, step=self.global_step)

            if self.checkpoint_metric == updated_key:
                self.is_best_model = True

        if self.is_best_model and not self.delayed_save_best_model:
            # if not delayed_save_best_model save, then save checkpoint
            self.save_best_checkpoint(global_step=self.global_step)

    def recover_checkpoint(self, checkpoint: Dict[str, Any], recover_optimizer: bool = True,
                           recover_train_progress: bool = True) -> None:
        self.recover_metrics(checkpoint=checkpoint)

        self.estimator.load_checkpoint(checkpoint=checkpoint,
                                       recover_optimizer=recover_optimizer,
                                       recover_train_progress=recover_train_progress)

    def recover_metrics(self, checkpoint: Dict[str, Any]) -> None:
        if "monitor_state" in checkpoint:
            monitor_state = checkpoint["monitor_state"]
        else:
            # for backward compatibility
            monitor_state = {
                "validation/mean_acc": checkpoint.get("best_val_acc", -np.inf),
                "test/mean_acc": checkpoint.get("best_test_acc", -np.inf),
            }

        self.monitor.load_state_dict(monitor_state)

    def cleanup_checkpoints(self) -> None:
        if not self.is_remove_old_checkpoint:
            # do nothing if the model do not save latest checkpoints or if all checkpoints are kept
            return

        checkpoint_paths = find_all_files(checkpoint_dir=self.log_dir,
                                          search_pattern=self.latest_checkpoint_pattern)

        # sort by recency (largest step first)
        checkpoint_paths.sort(key=lambda x: int(re.search(self.latest_checkpoint_pattern, x.name).group(1)),
                              reverse=True)

        # remove old checkpoints
        for checkpoint_path in checkpoint_paths[self.num_latest_checkpoints_kept:]:
            print(f"Removing old checkpoint \"{checkpoint_path}\"", flush=True)
            checkpoint_path.unlink()
