import torch

from utils import get_args, timing, set_random_seed, get_device, get_dataset
from models import get_augmenter
from simple_estimator import SimPLEEstimator
from ablation_estimator import AblationEstimator
from trainer import Trainer

# for type hint
from typing import Optional, Type
from argparse import Namespace
from torch.nn import Module

from utils.dataset import SSLDataModule


def get_estimator_type(estimator_type: str) -> Type[SimPLEEstimator]:
    if estimator_type == "ablation":
        return AblationEstimator
    else:
        return SimPLEEstimator


def get_estimator(args: Namespace,
                  augmenter: Module,
                  strong_augmenter: Module,
                  val_augmenter: Module,
                  num_classes: int,
                  in_channels: int,
                  device: Optional[torch.device],
                  args_override: Optional[Namespace] = None,
                  estimator_type: Optional[type(SimPLEEstimator)] = None):
    if estimator_type is None:
        estimator_type = get_estimator_type(args.estimator)

    if args.checkpoint_path is not None and \
            (bool(args.logger_config_dict.get("resume", False)) or not args.use_pretrain):
        estimator = estimator_type.from_checkpoint(
            augmenter=augmenter,
            strong_augmenter=strong_augmenter,
            val_augmenter=val_augmenter,
            checkpoint_path=args.checkpoint_path,
            num_classes=num_classes,
            in_channels=in_channels,
            device=device,
            args_override=args_override,
            recover_train_progress=True,
            recover_random_state=True)
    else:
        estimator = estimator_type(
            args,
            augmenter=augmenter,
            strong_augmenter=strong_augmenter,
            val_augmenter=val_augmenter,
            num_classes=num_classes,
            in_channels=in_channels,
            device=device)

    return estimator


@timing
def main(args: Namespace, datamodule: Optional[SSLDataModule] = None, device: Optional[torch.device] = None):
    if device is None:
        device = get_device(args.device)

    if datamodule is None:
        datamodule = get_dataset(args)

    # dataset stats
    dataset_mean = datamodule.dataset_mean
    dataset_std = datamodule.dataset_std
    image_size = datamodule.dims[1:]

    # build augmenters
    augmenter = get_augmenter(args.augmenter_type, image_size=image_size,
                              dataset_mean=dataset_mean, dataset_std=dataset_std)
    strong_augmenter = get_augmenter(args.strong_augmenter_type, image_size=image_size,
                                     dataset_mean=dataset_mean, dataset_std=dataset_std)
    val_augmenter = get_augmenter("validation", image_size=image_size, dataset_mean=dataset_mean,
                                  dataset_std=dataset_std)

    estimator = get_estimator(args,
                              augmenter=augmenter,
                              strong_augmenter=strong_augmenter,
                              val_augmenter=val_augmenter,
                              num_classes=datamodule.num_classes,
                              in_channels=datamodule.dims[0],
                              device=device,
                              args_override=args)
    trainer = Trainer(estimator, datamodule=datamodule)

    # training
    trainer.fit()

    # load the best state
    best_checkpoint_path = trainer.saver.find_best_checkpoint_path(ignore_absolute_best=False)

    if best_checkpoint_path is not None:
        best_checkpoint = torch.load(str(best_checkpoint_path), map_location=device)
        trainer.load_checkpoint(best_checkpoint)

    # evaluation
    trainer.test()


if __name__ == '__main__':
    parsed_args = get_args()

    # fix random seed
    set_random_seed(parsed_args.seed, is_cudnn_deterministic=parsed_args.debug_mode)

    main(parsed_args)
