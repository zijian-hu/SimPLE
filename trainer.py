import torch
from torch import distributed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pathlib import Path

from simple_estimator import SimPLEEstimator
from models.utils import get_weight_norm, get_gradient_norm, set_model_mode
from utils import timing, get_logger
from utils.dataset import get_batch
from checkpoint_saver import CheckpointSaver

# for type hint
from typing import Optional, Dict, Any, Union, Tuple
from argparse import Namespace
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from utils.dataset import SSLDataModule
from loss.types import LossInfoType
from utils.types import BatchGeneratorType
from models.types import LRSchedulerType

IS_DISTRIBUTED_AVAILABLE = distributed.is_available()


class Trainer:
    def __init__(self, estimator: SimPLEEstimator, datamodule: SSLDataModule):
        # data loaders
        self._test_loader: Optional[DataLoader] = None
        self._validation_loader: Optional[DataLoader] = None
        self._labeled_train_loader: Optional[DataLoader] = None
        self._unlabeled_train_loader: Optional[DataLoader] = None

        # assign estimator and intermediate variables
        self._estimator = estimator

        self.exp_args = self.estimator.exp_args
        self.use_ema = self.estimator.use_ema
        self.log_interval = self.estimator.log_interval

        # setup dataset
        self.datamodule = datamodule

        # setup distributed training
        if self.is_distributed:
            self.to_distributed_data_parallel()

        # setup logger
        if not self.is_main_thread:
            # if not in main thread, use no-op logger (i.e. no logging)
            self.exp_args.logger = "nop"

        self.logger = get_logger(self.exp_args)
        self.logger.watch(self.get_model(), log='all')

        # setup checkpoint saver
        self.saver = CheckpointSaver(estimator=self.estimator,
                                     logger=self.logger,
                                     checkpoint_metric="validation/mean_acc",
                                     best_checkpoint_str="best@step-{global_step}.pth",
                                     best_checkpoint_pattern=r"best@step-(\d+)\.pt",
                                     latest_checkpoint_str="latest@step-{global_step}.pth",
                                     latest_checkpoint_pattern=r"latest@step-(\d+)\.pt",
                                     delayed_best_model_saving=True)

    @property
    def exp_args(self) -> Namespace:
        return self._exp_args

    @exp_args.setter
    def exp_args(self, exp_args: Namespace) -> None:
        self._exp_args = exp_args

        # assign intermediate variables
        self.log_labeled_train_set = self.exp_args.log_labeled_train_set
        self.log_unlabeled_train_set = self.exp_args.log_unlabeled_train_set
        self.log_validation_set = self.exp_args.log_validation_set
        self.log_test_set = self.exp_args.log_test_set

        if self.is_main_thread:
            log_dir_path = Path(self.exp_args.log_dir)

            if not log_dir_path.exists():
                log_dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def estimator(self) -> SimPLEEstimator:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: SimPLEEstimator) -> None:
        self._estimator = estimator

        # update intermediate variables
        self.exp_args = self.estimator.exp_args
        self.use_ema = self.estimator.use_ema
        self.log_interval = self.estimator.log_interval

        # setup distributed training
        if self.is_distributed:
            self.to_distributed_data_parallel()

        # update saver
        self.saver.estimator = self.estimator

    @property
    def datamodule(self) -> SSLDataModule:
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: SSLDataModule) -> None:
        self._datamodule = datamodule

        # setup datasets
        if self.is_main_thread:
            self.datamodule.prepare_data()
        self.datamodule.setup()
        # save dataset split info
        if self.is_main_thread:
            self.datamodule.save_split_info(self.exp_args.log_dir)

        # assign data loaders
        self.labeled_train_loader, self.unlabeled_train_loader = self.datamodule.train_dataloader()
        self.validation_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()

    @property
    def model(self) -> Module:
        return self.estimator.model

    @model.setter
    def model(self, model: Module) -> None:
        self.estimator.model = model

    @property
    def optimizer(self) -> Optimizer:
        return self.estimator.optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self.estimator.optimizer = optimizer

    @property
    def lr_scheduler(self) -> LRSchedulerType:
        return self.estimator.lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler: LRSchedulerType) -> None:
        self.estimator.lr_scheduler = lr_scheduler

    @property
    def test_loader(self) -> Optional[DataLoader]:
        return self._test_loader

    @test_loader.setter
    def test_loader(self, loader: DataLoader) -> None:
        self._test_loader = loader

    @property
    def validation_loader(self) -> Optional[DataLoader]:
        return self._validation_loader

    @validation_loader.setter
    def validation_loader(self, loader: DataLoader) -> None:
        self._validation_loader = loader

    @property
    def labeled_train_loader(self) -> Optional[DataLoader]:
        return self._labeled_train_loader

    @labeled_train_loader.setter
    def labeled_train_loader(self, loader: DataLoader) -> None:
        self._labeled_train_loader = loader

    @property
    def unlabeled_train_loader(self) -> Optional[DataLoader]:
        return self._unlabeled_train_loader

    @unlabeled_train_loader.setter
    def unlabeled_train_loader(self, loader: DataLoader) -> None:
        self._unlabeled_train_loader = loader

    @property
    def device(self) -> torch.device:
        return self.estimator.device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.estimator.device = device

    @property
    def num_epochs(self) -> int:
        return self.estimator.num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int) -> None:
        assert num_epochs >= 1

        old_num_epochs = self.num_epochs
        self.estimator.num_epochs = num_epochs

        if old_num_epochs != num_epochs:
            # update configs
            self.logger.log({"num_epochs": num_epochs}, is_config=True)

    @property
    def num_warmup_epochs(self) -> int:
        return self.estimator.num_warmup_epochs

    @num_warmup_epochs.setter
    def num_warmup_epochs(self, num_warmup_epochs: int) -> None:
        assert num_warmup_epochs >= 0

        old_num_warmup_epochs = self.num_warmup_epochs
        self.estimator.num_warmup_epochs = num_warmup_epochs

        if old_num_warmup_epochs != num_warmup_epochs:
            # update configs
            self.logger.log({"num_warmup_epochs": num_warmup_epochs}, is_config=True)

    @property
    def num_step_per_epoch(self) -> int:
        return self.estimator.num_step_per_epoch

    @num_step_per_epoch.setter
    def num_step_per_epoch(self, num_step_per_epoch: int) -> None:
        assert num_step_per_epoch >= 1

        old_num_step_per_epoch = self.num_step_per_epoch
        self.estimator.num_step_per_epoch = num_step_per_epoch

        if old_num_step_per_epoch != num_step_per_epoch:
            # update configs
            self.logger.log({"num_step_per_epoch": num_step_per_epoch}, is_config=True)

    @property
    def max_eval_step(self) -> int:
        return self.exp_args.max_eval_step

    @max_eval_step.setter
    def max_eval_step(self, max_eval_step: int) -> None:
        assert max_eval_step >= 1

        old_max_eval_step = self.max_eval_step
        self.estimator.max_eval_step = max_eval_step

        if old_max_eval_step != max_eval_step:
            # update configs
            self.logger.log({"max_eval_step": max_eval_step}, is_config=True)

    @property
    def global_step(self) -> int:
        return self.estimator.global_step

    @global_step.setter
    def global_step(self, global_step: int) -> None:
        self.estimator.global_step = global_step

    @property
    def epoch(self) -> int:
        return self.estimator.epoch

    @property
    def is_main_thread(self) -> bool:
        """
        Process with on rank 0 node and with local_rank 0 is considered the main thread.
        This flag is used in distributed training where only main thread save checkpoints and log

        Returns: True if the current thread is the main thread
        """
        return not self.is_distributed or (self.exp_args.local_rank <= 0 and distributed.get_rank() == 0)

    @property
    def is_distributed(self) -> bool:
        return IS_DISTRIBUTED_AVAILABLE and distributed.is_initialized()

    def get_model(self) -> Module:
        return self.estimator.get_model()

    def get_trainable_model(self) -> Module:
        return self.estimator.get_trainable_model()

    @timing
    def fit(self, is_update_best_checkpoint: bool = True) -> None:
        start_epoch = self.epoch

        for epoch in range(start_epoch, self.num_epochs):
            train_log_info = self.training_epoch()

            # evaluate model and log stats
            self.eval_and_log(train_log_info)

            # save the latest checkpoint
            self.save_latest_checkpoint()

        if is_update_best_checkpoint:
            # update best metrics and save a new checkpoint
            self.update_best_checkpoint()

    def test(self) -> None:
        # testing
        self.model.eval()

        log_info = self.validation_epoch(self.test_loader, desc="Test")

        # save test result
        self.logger.log(
            log_info=log_info,
            prefix="test/final",
            is_summary=True)

        # TODO: add best metric update

    @timing
    def training_epoch(self) -> Dict[str, Any]:
        # turn on training mode
        self.model.train()

        # cleanup aggregator
        self.logger.reset_aggregator()

        # when global_step is recovered from checkpoint, it might not be the start/end of an epoch
        total_step = self.num_step_per_epoch - (self.global_step % self.num_step_per_epoch)

        # create progress bar
        p_bar = tqdm(self.get_train_batch(max_iter=total_step),
                     desc=f"Epoch {self.epoch + 1}",
                     total=total_step)

        for batch_idx, batch in p_bar:
            # compute train loss
            loss, log_dict = self.training_step(batch, batch_idx)

            # update parameters
            self.update_model_params(loss)

            # update logs
            self.logger.accumulate_log(log_info=log_dict["log"], plot_info=log_dict["plot"])

            # increment global step
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                outputs = self.logger.aggregate_log(reduction="mean")
                self.log_train_info(outputs, is_commit=True, prefix="train", step=self.global_step)

                # reset output history
                self.logger.reset_aggregator()

            # display loss in progress bar
            p_bar.set_postfix(loss=log_dict["log"]['loss'])

        outputs = self.logger.aggregate_log(reduction="mean")
        outputs = self.log_train_info(outputs, is_commit=False)

        self.training_epoch_end()

        return outputs

    def training_epoch_end(self, *args, **kwargs) -> None:
        self.estimator.training_epoch_end(*args, **kwargs)

    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_idx: int) -> Tuple[Tensor, LossInfoType]:
        loss, log_dict = self.estimator.training_step(batch, batch_idx)

        return loss, {k: self.reduce_log_dict(v) for k, v in log_dict.items()}

    def validation_epoch(self,
                         data_loader: DataLoader,
                         desc: str = "",
                         max_val_iter: Optional[int] = None) -> Dict[str, Any]:
        if max_val_iter is None:
            max_val_iter = int(min(len(data_loader), self.max_eval_step))
        max_val_iter = min(len(data_loader), max_val_iter)

        # cleanup aggregator
        self.logger.reset_aggregator()

        with set_model_mode(self.model, mode=False):
            with torch.no_grad():
                # create progress bar
                p_bar = tqdm(self.get_eval_batch(data_loader, max_iter=max_val_iter), desc=desc, total=max_val_iter)

                for batch_idx, batch in p_bar:
                    log_info = self.validation_step(batch, batch_idx)
                    self.logger.accumulate_log(log_info=log_info)

                    # display loss in progress bar
                    p_bar.set_postfix(loss=log_info["loss"])

        return self.logger.aggregate_log(reduction="mean")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        log_dict = self.estimator.validation_step(batch, batch_idx)

        return self.reduce_log_dict(log_dict)

    def update_model_params(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()

        # backprop
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.get_trainable_model().parameters(),
                                       max_norm=self.estimator.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        # model EMA
        if self.use_ema:
            self.get_model().update()

    def log_train_info(self, log_info: Dict[str, Any], is_commit: bool = False, **log_kwargs) -> Dict[str, Any]:
        if len(log_info) > 0:
            # log gradient norm and weight norm
            model_info_dict = {
                "gradient_norm": get_gradient_norm(self.model, grad_enabled=False)
            }
            if self.use_ema:
                model_info_dict["weight_norm"] = get_weight_norm(self.get_model().model, grad_enabled=False)
                model_info_dict["shadow_weight_norm"] = get_weight_norm(self.get_model().shadow, grad_enabled=False)
            else:
                model_info_dict["weight_norm"] = get_weight_norm(self.model, grad_enabled=False)

            log_info.update(self.reduce_log_dict(model_info_dict))

            if is_commit:
                self.logger.log(log_info, **log_kwargs)

        return log_info

    def eval_and_log(self, train_log_info: Optional[Dict[str, Any]] = None) -> None:
        if train_log_info is None:
            train_log_info = dict()

        if self.log_labeled_train_set:
            # log train loss and accuracy
            log_info = self.validation_epoch(self.labeled_train_loader, desc="Labeled")
            log_info.pop("loss")
            self.logger.log(log_info, prefix="train", log_info_override=train_log_info, step=self.global_step)

        if self.log_unlabeled_train_set:
            # log unlabeled train loss and accuracy
            log_info = self.validation_epoch(self.unlabeled_train_loader, desc="Unlabeled")
            self.logger.log(log_info, prefix="unlabeled", step=self.global_step)

        # log validation loss and accuracy
        if self.log_validation_set:
            log_info = self.validation_epoch(self.validation_loader, desc="Validation")
            self.logger.log(log_info, prefix="validation", step=self.global_step)

        if self.log_test_set:
            # log test loss and accuracy
            log_info = self.validation_epoch(self.test_loader, desc="Test")
            self.logger.log(log_info, prefix="test", step=self.global_step)

    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        checkpoint_path: str,
                        is_logger_save: bool = False) -> None:
        """
        Save checkpoint

        Args:
            checkpoint: The checkpoint object. If None, save the current estimator.
            checkpoint_path: Path to save the checkpoint.
            is_logger_save: If True, also save the model through logger

        Returns:

        """
        if self.is_main_thread:
            self.saver.save_checkpoint(checkpoint=checkpoint,
                                       checkpoint_path=checkpoint_path,
                                       is_logger_save=is_logger_save)

    def save_best_checkpoint(self,
                             checkpoint: Optional[Dict[str, Any]] = None,
                             is_logger_save: bool = False) -> None:
        if self.is_main_thread:
            self.saver.save_best_checkpoint(
                checkpoint=checkpoint,
                is_logger_save=is_logger_save,
                global_step=self.global_step)

    def save_latest_checkpoint(self,
                               checkpoint: Optional[Dict[str, Any]] = None,
                               is_logger_save: bool = False) -> None:
        if self.is_main_thread:
            self.saver.save_latest_checkpoint(
                checkpoint=checkpoint,
                is_logger_save=is_logger_save,
                global_step=self.global_step)

    def update_best_checkpoint(self) -> None:
        """
        Update the logged metrics for the best checkpoint

        Returns:

        """
        if self.is_main_thread:
            self.saver.update_best_checkpoint()

        if self.is_distributed:
            # sync all processes, wait until best checkpoint is updated
            distributed.barrier()

    def load_checkpoint(self, checkpoint: Dict[str, Any], recover_optimizer: bool = True,
                        recover_train_progress: bool = True) -> None:
        self.saver.recover_checkpoint(checkpoint=checkpoint,
                                      recover_optimizer=recover_optimizer,
                                      recover_train_progress=recover_train_progress)

    def reduce_tensor(self, inputs: Tensor) -> Tensor:
        outputs = inputs.detach().clone()

        if not self.is_distributed:
            # by default, windows do not support torch.distributed
            return outputs

        distributed.all_reduce(outputs, op=distributed.ReduceOp.SUM)
        outputs /= distributed.get_world_size()
        return outputs

    def reduce_log_dict(self, log_dict: Dict[str, Union[Any, Tensor]]) -> Dict[str, Union[Any, Tensor]]:
        return {k: self.reduce_tensor(v).item() if isinstance(v, Tensor) else v for k, v in log_dict.items()}

    def get_train_batch(self, max_iter: int) -> BatchGeneratorType:
        if self.is_distributed:
            # set sampler epoch when distributed
            self.sampler_set_epoch(self.labeled_train_loader, self.epoch)
            self.sampler_set_epoch(self.unlabeled_train_loader, self.epoch)

        return self.datamodule.get_train_batch([self.labeled_train_loader, self.unlabeled_train_loader],
                                               max_iter=max_iter,
                                               is_repeat=True)

    def get_eval_batch(self, data_loader: DataLoader, max_iter: int) -> BatchGeneratorType:
        if self.is_distributed:
            # set sampler epoch when distributed
            self.sampler_set_epoch(data_loader, self.epoch)

        return get_batch([data_loader],
                         max_iter=max_iter,
                         is_repeat=False)

    def to_data_parallel(self) -> None:
        print(f"Run in data parallel w/ {torch.cuda.device_count()} GPUs")

        from torch.nn.parallel import DataParallel
        if self.use_ema:
            self.model.model = DataParallel(self.model.model)
            self.model.shadow = DataParallel(self.model.shadow)
        else:
            self.model = DataParallel(self.model)

        self.estimator.augmenter = DataParallel(self.estimator.augmenter)
        self.estimator.strong_augmenter = DataParallel(self.estimator.strong_augmenter)
        self.estimator.val_augmenter = DataParallel(self.estimator.val_augmenter)

    def to_distributed_data_parallel(self) -> None:
        print(f"Run in distributed data parallel w/ {torch.cuda.device_count()} GPUs")

        # setup models
        from torch.nn.parallel import DistributedDataParallel
        from torch.nn import SyncBatchNorm

        local_rank = self.exp_args.local_rank

        self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True)

        # setup data loaders
        self.test_loader = self.to_distributed_loader(
            self.test_loader,
            shuffle=False,
            num_workers=self.exp_args.num_workers,
            pin_memory=True,
            drop_last=False)

        self.validation_loader = self.to_distributed_loader(
            self.validation_loader,
            shuffle=False,
            num_workers=self.exp_args.num_workers,
            pin_memory=True,
            drop_last=True)

        self.labeled_train_loader = self.to_distributed_loader(
            self.labeled_train_loader,
            shuffle=True,
            num_workers=self.exp_args.num_workers,
            pin_memory=True,
            drop_last=True)

        self.unlabeled_train_loader = self.to_distributed_loader(
            self.unlabeled_train_loader,
            shuffle=True,
            num_workers=self.exp_args.num_workers,
            pin_memory=True,
            drop_last=True)

    @staticmethod
    def to_distributed_loader(loader: DataLoader, shuffle: bool, **kwargs) -> Optional[DataLoader]:
        return DataLoader(dataset=loader.dataset,
                          batch_size=loader.batch_size,
                          sampler=DistributedSampler(loader.dataset, shuffle=shuffle),
                          **kwargs) if loader is not None else None

    @staticmethod
    def sampler_set_epoch(loader: DataLoader, epoch: int) -> None:
        sampler = loader.sampler

        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
