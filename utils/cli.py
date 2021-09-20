import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np

from .utils import str_to_bool, filter_dict
from .file_io import read_yaml

# for type hint
from typing import Union, Tuple, List, Optional, Set, Dict, Any
from argparse import Namespace, ArgumentParser

ArgsOutType = Union[Namespace, Tuple[Namespace, List[str]]]

# choices for CLI args. the first element should be default choice
AUGMENTER_TYPE_CHOICES = ["simple", "randaugment", "fixed"]

LOGGER_EXCLUDED_KEYS = {
    "local_rank",
    "logger_run_name",
    "tags",
    "notes",
    "dry_run",
    "is_display_plots",
    "wandb_resume",
    "logger_config_path",
    "logger_config_dict",
}


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SimPLE", fromfile_prefix_chars="@")

    # Experiment
    parser.add_argument('-e',
                        '--estimator',
                        '--estimator-type',
                        dest="estimator",
                        choices=["default", "simple", "ablation"],
                        default="default",
                        type=str,
                        help=f"Estimator type")

    # Network params
    parser.add_argument('--num-epochs',
                        dest="num_epochs",
                        default=1024,
                        type=int,
                        help='number of total training epochs')

    parser.add_argument('--num-warmup-epochs',
                        '--num-warm-up-epochs',
                        dest="num_warmup_epochs",
                        default=None,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss linear ramp-up during training'
                             'set to 0 to disable linear ramp-up')

    parser.add_argument('--num-step-per-epoch',
                        dest="num_step_per_epoch",
                        default=1024,
                        type=int,
                        help='number of training steps for each epoch')

    parser.add_argument('--batch-size',
                        '--train-batch-size',
                        '--labeled-train-batch-size',
                        dest="train_batch_size",
                        default=64,
                        type=int,
                        help='train batch size')

    parser.add_argument('--unlabeled-batch-size',
                        '--unlabeled-train-batch-size',
                        dest="unlabeled_train_batch_size",
                        default=None,
                        type=int,
                        help='unlabeled train batch size')

    parser.add_argument('--test-batch-size',
                        dest="test_batch_size",
                        default=None,
                        type=int,
                        help='test batch size')

    parser.add_argument('--lr',
                        '--learning-rate',
                        dest="learning_rate",
                        default=0.002,
                        type=float,
                        help='learning rate')

    parser.add_argument('--feature-lr',
                        '--feature-learning-rate',
                        dest="feature_learning_rate",
                        default=None,
                        type=float,
                        help='feature extractor learning rate')

    parser.add_argument('--lr-step-size',
                        '--learning-rate-step-size',
                        dest="lr_step_size",
                        default=5,
                        type=int,
                        help='step size for step learning rate decay')

    parser.add_argument('--lr-gamma',
                        '--learning-rate-gamma',
                        dest="lr_gamma",
                        default=0.1,
                        type=float,
                        help='factor for step learning rate decay')

    parser.add_argument('--lr-cosine-factor',
                        '--learning-rate-cosine-factor',
                        dest="lr_cosine_factor",
                        default=0.49951171875,
                        type=float,
                        help='factor for cosine learning rate decay')

    parser.add_argument('--lr-warmup-step',
                        '--lr-warmup-steps',
                        '--learning-rate-warmup-step',
                        '--learning-rate-warmup-steps',
                        dest="lr_warmup_step",
                        default=0,
                        type=int,
                        help='number of warmup steps for cosine learning rate decay with warmup')

    parser.add_argument('--optimizer-momentum',
                        dest="optimizer_momentum",
                        default=0.9,
                        type=float,
                        help='momentum for optimizer')

    parser.add_argument('--max-eval-step',
                        dest="max_eval_step",
                        default=np.inf,
                        type=int,
                        help='max evaluation step for each evaluation set')

    parser.add_argument('--weight-decay',
                        dest="weight_decay",
                        default=0.02,
                        type=float,
                        help='weight decay')

    parser.add_argument('--ema-decay',
                        dest="ema_decay",
                        default=0.999,
                        type=float,
                        help='Decay value for exponential moving average (EMA) of parameters. '
                             'Set to 0 to disable EMA')

    # MixMatch params
    parser.add_argument('--K',
                        dest="k",
                        default=2,
                        type=int,
                        help="number of augmentations for MixMatch")

    parser.add_argument('--K-strong',
                        dest="k_strong",
                        default=None,
                        type=int,
                        help="number of augmentations for pair MixMatch")

    parser.add_argument('--T',
                        dest="t",
                        default=0.5,
                        type=float,
                        help="sharpening temperature for MixMatch")

    parser.add_argument('--alpha',
                        dest="alpha",
                        default=0.75,
                        type=float,
                        help="Beta distribution parameter for MixMatch")

    parser.add_argument('--lambda-u',
                        dest="lambda_u",
                        default=75.,
                        type=float,
                        help="weight for unlabeled loss")

    parser.add_argument('--mixmatch-type',
                        dest="mixmatch_type",
                        choices=["mixmatch", "enhanced", "simple"],
                        default=None,
                        type=str,
                        help=f"mixmatch type")

    parser.add_argument('--model',
                        '--model-type',
                        dest="model_type",
                        choices=["wrn28-2", "wrn28-8", "resnet18", "resnet50"],
                        default="wrn28-2",
                        type=str,
                        help=f"model type")

    parser.add_argument('--optimizer',
                        '--optimizer-type',
                        dest="optimizer_type",
                        choices=["adamw", "sgd"],
                        default="adamw",
                        type=str,
                        help=f"optimizer type")

    parser.add_argument('--lr-scheduler',
                        '--lr-scheduler-type',
                        dest="lr_scheduler_type",
                        choices=["nop", "cosine_decay", "cosine_warmup_decay", "step_decay"],
                        default="nop",
                        type=str,
                        help=f"learning rate scheduler type")

    parser.add_argument('--augmenter-type',
                        dest="augmenter_type",
                        choices=AUGMENTER_TYPE_CHOICES,
                        default=AUGMENTER_TYPE_CHOICES[0],
                        type=str,
                        help=f"augmenter type")

    parser.add_argument('--strong-augmenter-type',
                        dest="strong_augmenter_type",
                        choices=AUGMENTER_TYPE_CHOICES,
                        default=AUGMENTER_TYPE_CHOICES[1],
                        type=str,
                        help=f"strong augmenter type")

    parser.add_argument('--ema-type',
                        dest="ema_type",
                        choices=["default", "full"],
                        default="default",
                        type=str,
                        help=f"EMA type")

    parser.add_argument('--rampup-type',
                        '--ramp-up-type',
                        dest="ramp_up_type",
                        choices=["linear"],
                        default="linear",
                        type=str,
                        help=f"Loss ramp-up type")

    parser.add_argument('--u-loss-type',
                        '--unsupervised-loss-type',
                        dest="u_loss_type",
                        choices=["mse", "entropy"],
                        default="mse",
                        type=str,
                        help=f"Unsupervised loss type")

    parser.add_argument('--u-loss-thresholded',
                        '--unsupervised-loss-thresholded',
                        dest="u_loss_thresholded",
                        default=False,
                        action='store_true',
                        help='Apply threshold to unsupervised loss')

    # pair loss settings
    parser.add_argument('--lambda-pair',
                        dest="lambda_pair",
                        default=0.,
                        type=float,
                        help="weight for pair loss")

    parser.add_argument('--sim-type',
                        '--similarity-type',
                        dest="similarity_type",
                        choices=["bhc"],
                        default="bhc",
                        type=str,
                        help=f"similarity function type for pair loss")

    parser.add_argument('--dist-type',
                        '--distance-type',
                        '--dist-loss-type',
                        '--distance-loss-type',
                        dest="distance_loss_type",
                        choices=["bhc", "entropy", "l2"],
                        default="bhc",
                        type=str,
                        help=f"statistical distance function type for pair loss")

    parser.add_argument('--conf-threshold',
                        '--confidence-threshold',
                        dest="confidence_threshold",
                        default=0.,
                        type=float,
                        help="confidence threshold for pair loss and unsupervised loss")

    parser.add_argument('--sim-threshold',
                        '--similarity-threshold',
                        dest="similarity_threshold",
                        default=0.,
                        type=float,
                        help="similarity threshold for pair loss to be pushed together")

    # train settings
    parser.add_argument("--max-grad-norm",
                        dest="max_grad_norm",
                        type=float,
                        default=np.inf,
                        help="max gradient norm allowed (used for gradient clipping)")

    # ablation settings
    parser.add_argument('--use-ema',
                        dest="use_ema",
                        default=True,
                        action='store_true',
                        help='Turn on exponential moving average of params.')

    parser.add_argument('--no-ema',
                        dest="use_ema",
                        default=True,
                        action='store_false',
                        help='Turn off exponential moving average of params.')

    # data and dataset settings
    parser.add_argument('--labeled-train-size',
                        dest="labeled_train_size",
                        default=4_000,
                        type=int,
                        help='number of labeled training data')

    parser.add_argument('--validation-size',
                        dest="validation_size",
                        default=5_000,
                        type=int,
                        help='number of validation data')

    parser.add_argument('--data-dir',
                        dest="data_dir",
                        default="data",
                        type=str,
                        help="path to data directory")

    parser.add_argument('--dataset',
                        dest="dataset",
                        choices=["cifar10", "cifar100", "miniimagenet", "svhn", "domainnet-real"],
                        required=True,
                        type=str,
                        help=f"dataset type")

    parser.add_argument('--dims',
                        '--data-dims',
                        dest="data_dims",
                        default=None,
                        type=int,
                        nargs=3,
                        help='Data dimensions in CHW')

    parser.add_argument('--num-workers',
                        dest="num_workers",
                        default=0,
                        type=int,
                        help="number of processes for data loader")

    parser.add_argument('--checkpoint-path',
                        dest="checkpoint_path",
                        default=None,
                        type=str,
                        help="checkpoint path to recover from")

    parser.add_argument('--use-pretrain',
                        dest="use_pretrain",
                        default=False,
                        action='store_true',
                        help='Set to True to use pre-trained model')

    # evaluation settings
    parser.add_argument('--log-labeled-train-set',
                        '--enable-labeled-train-log',
                        dest="log_labeled_train_set",
                        default=True,
                        action='store_true',
                        help='Enable labeled train set logging and evaluation at the end of every train epoch')

    parser.add_argument('--no-labeled-train-log',
                        '--no-log-labeled-train-set',
                        '--disable-labeled-train-log',
                        dest="log_labeled_train_set",
                        default=True,
                        action='store_false',
                        help='Disable labeled train set logging and evaluation at the end of every train epoch')

    parser.add_argument('--log-unlabeled-train-set',
                        '--enable-unlabeled-train-log',
                        dest="log_unlabeled_train_set",
                        default=True,
                        action='store_true',
                        help='Enable unlabeled train set logging and evaluation at the end of every train epoch')

    parser.add_argument('--no-unlabeled-train-log',
                        '--no-log-unlabeled-train-set',
                        '--disable-unlabeled-train-log',
                        dest="log_unlabeled_train_set",
                        default=True,
                        action='store_false',
                        help='Disable unlabeled train set logging and evaluation at the end of every train epoch')

    parser.add_argument('--log-validation-set',
                        '--enable-validation-log',
                        dest="log_validation_set",
                        default=True,
                        action='store_true',
                        help='Enable validation set logging and evaluation after every at the end of every train '
                             'epoch')

    parser.add_argument('--no-validation-log',
                        '--no-log-validation-set',
                        '--disable-validation-log',
                        dest="log_validation_set",
                        default=True,
                        action='store_false',
                        help='Disable validation set logging and evaluation after every at the end of every train '
                             'epoch')

    parser.add_argument('--log-test-set',
                        '--enable-test-log',
                        dest="log_test_set",
                        default=True,
                        action='store_true',
                        help='Disable test set logging and evaluation after every at the end of every train epoch')

    parser.add_argument('--no-test-log',
                        '--no-log-test-set',
                        '--disable-test-log',
                        dest="log_test_set",
                        default=True,
                        action='store_false',
                        help='Disable test set logging and evaluation after every at the end of every train epoch')

    # other settings
    parser.add_argument('--device',
                        dest="device",
                        default='0',
                        type=str,
                        help='"cpu" if using CPU or id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--seed',
                        dest="seed",
                        type=int,
                        default=1234,
                        help='Random seed')

    parser.add_argument('--cudnn-benchmark',
                        '--use-cudnn-benchmark',
                        '--enable-cudnn-benchmark',
                        dest="cudnn_benchmark",
                        default=True,
                        action='store_true',
                        help='Turn on cuDNN benchmark')

    parser.add_argument('--no-cudnn-benchmark',
                        '--disable-cudnn-benchmark',
                        dest="cudnn_benchmark",
                        default=True,
                        action='store_false',
                        help='Turn off cuDNN benchmark')

    parser.add_argument('--debug-mode',
                        '--enable-debug-mode',
                        dest="debug_mode",
                        default=False,
                        action='store_true',
                        help='Turn on debugging mode')

    parser.add_argument('--fast-dev-mode',
                        '--enable-fast-dev-mode',
                        dest="fast_dev_mode",
                        default=False,
                        action='store_true',
                        help='Turn on debugging mode')

    # multi GPU settings
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help='node rank for distributed training')

    # log settings
    parser.add_argument('--log-dir',
                        dest="log_dir",
                        default=None,
                        type=str,
                        help="path to log directory")

    parser.add_argument('--logger',
                        dest="logger",
                        choices=["print", "wandb", "nop"],
                        default="print",
                        type=str,
                        help=f"Logger type")

    parser.add_argument('--log-interval',
                        dest="log_interval",
                        type=int,
                        default=None,
                        help=f"interval for logging train loss")

    parser.add_argument('--logger-run-name',
                        dest="logger_run_name",
                        type=str,
                        default=None,
                        help=f"name for the current execution")

    parser.add_argument("--tags",
                        dest="tags",
                        type=str,
                        nargs="+",
                        default=None,
                        help=f"tags for wandb logger")

    parser.add_argument("--notes",
                        dest="notes",
                        type=str,
                        default=None,
                        help=f"notes for wandb logger")

    parser.add_argument('--logger-config-path',
                        dest="logger_config_path",
                        default=None,
                        type=str,
                        help="path to logger configuration file (only support \".yaml\" and \".yml\" format)")

    parser.add_argument('--dryrun',
                        '--dry-run',
                        dest="dry_run",
                        default=False,
                        action='store_true',
                        help='Disable syncing to wandb')

    parser.add_argument('--wandb-resume',
                        dest="wandb_resume",
                        type=str,
                        default=False,
                        help='if set to True, the run auto resumes; can also be a unique string for manual resuming')

    parser.add_argument('--display-plot',
                        '--display-plots',
                        '--enable-plot-display',
                        dest="is_display_plots",
                        default=False,
                        action='store_true',
                        help='Enable plot display')

    parser.add_argument('--disable-plot-display',
                        dest="is_display_plots",
                        default=False,
                        action='store_false',
                        help='Disable plot display')

    parser.add_argument('--num-latest-ckpt-kept',
                        '--num-latest-checkpoints-kept',
                        dest="num_latest_checkpoints_kept",
                        type=int,
                        default=None,
                        help=f"Number of latest checkpoints to be kept")

    return parser


def parse_args(args: Namespace) -> Namespace:
    # update estimator config
    if args.estimator == "simple":
        args.mixmatch_type = "simple"
        args.u_loss_thresholded = True

        if args.num_warmup_epochs is None:
            args.num_warmup_epochs = 0

    elif args.estimator == "ablation":
        args.lambda_u = 0.
        args.lambda_pair = 0.

    elif args.mixmatch_type is None:
        # if estimator type is not "simple" and mixmatch type is unspecified
        if args.lambda_pair != 0:
            args.mixmatch_type = "enhanced"
        else:
            args.mixmatch_type = "mixmatch"

    # update args values
    if args.unlabeled_train_batch_size is None:
        args.unlabeled_train_batch_size = args.train_batch_size

    if args.test_batch_size is None:
        args.test_batch_size = args.train_batch_size

    if args.num_warmup_epochs is None:
        args.num_warmup_epochs = args.num_epochs

    if args.fast_dev_mode:
        args.num_epochs = 2
        args.num_step_per_epoch = 10
        args.max_eval_step = 1

    if args.log_interval is None:
        # update args.log_interval
        args.log_interval = args.num_step_per_epoch

    if args.use_pretrain and args.feature_learning_rate is None:
        args.feature_learning_rate = args.learning_rate

    if args.k_strong is None:
        args.k_strong = args.k

    args.num_workers = max(args.num_workers, 0)

    # value check
    verify_args(args)

    if isinstance(args.wandb_resume, str):
        args.wandb_resume = str_to_bool(args.wandb_resume)

    # assign logger_config_dict to args
    args.logger_config_dict = parse_logger_config(args)

    if (args.checkpoint_path is not None) and (not args.use_pretrain) and \
            bool(args.logger_config_dict.get("resume", False)):
        # use the same log_dir if continuing on previous run
        args.log_dir = str(Path(args.checkpoint_path).parent)

    if args.log_dir is None:
        args.log_dir = generate_log_path(log_dir="logs", basename=args.logger_config_dict["name"])

    if args.cudnn_benchmark:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    return args


def parse_logger_config(args: Namespace) -> Dict[str, Union[Optional[str], bool, List[str]]]:
    # default parameters
    logger_config_dict: Dict[str, Union[Optional[str], bool, List[str]]] = dict(name=None)

    if args.logger == "wandb":
        # wandb default parameters
        logger_config_dict.update(dict(
            entity=None,
            project=None,
            tags=[],
            notes=None,
            resume=False,
            mode="offline"))

        if args.logger_config_path is not None:
            logger_config_path = Path(args.logger_config_path)

            if logger_config_path.is_file() and logger_config_path.suffix.lower() in {".yaml", ".yml"}:
                # load config file and update logger_config_dict
                logger_config_dict.update(read_yaml(logger_config_path))

        # overwrite logger_config_dict with CLI arguments
        if args.tags is not None:
            logger_config_dict["tags"] = args.tags

        if args.notes is not None:
            logger_config_dict["notes"] = args.notes

        if args.wandb_resume is not False:
            logger_config_dict["resume"] = args.wandb_resume

        if args.debug_mode and "debug" not in logger_config_dict["tags"]:
            logger_config_dict["tags"].append("debug")

        assert args.checkpoint_path is not None or not bool(logger_config_dict["resume"]), \
            f"checkpoint_path must be set if \"resume\" or \"wandb_resume\" is not False"

        if args.dry_run:
            logger_config_dict["mode"] = "offline"

    # overwrite logger_config_dict with CLI arguments
    if args.logger_run_name is not None:
        logger_config_dict["name"] = args.logger_run_name

    # generate experiment name for logger
    if logger_config_dict["name"] is None:
        logger_config_dict["name"] = generate_logger_run_name(args)

    return logger_config_dict


def verify_args(args: Namespace) -> None:
    assert args.num_epochs >= 1, f"num_epochs must be >= 1 but get {args.num_epochs}"
    assert args.num_warmup_epochs >= 0, f"num_warmup_epochs must be >= 0 but get {args.num_warmup_epochs}"
    assert args.num_step_per_epoch >= 1, f"num_step_per_epoch must be >= 1 but get {args.num_step_per_epoch}"
    assert args.train_batch_size >= 1, f"train_batch_size must be >= 1 but get {args.train_batch_size}"
    assert args.test_batch_size >= 1, f"test_batch_size must be >= 1 but get {args.test_batch_size}"
    assert args.learning_rate > 0, f"learning_rate must be > 0 but get {args.learning_rate}"
    if args.feature_learning_rate is not None:
        assert args.feature_learning_rate > 0, f"feature_learning_rate must be > 0 but get {args.feature_learning_rate}"
    assert args.weight_decay >= 0, f"weight_decay must be >= 0 but get {args.weight_decay}"
    assert args.max_eval_step >= 1, f"max_eval_step must be >= 1 but get {args.max_eval_step}"
    assert 0 <= args.ema_decay <= 1, f"ema_decay must be in [0, 1] but get {args.ema_decay}"
    assert args.k >= 1, f"K must be >= 1 but get {args.k}"
    assert args.k_strong >= 1, f"k_strong must be >= 1 but get {args.k_strong}"
    assert args.t != 0, f"T must not equal 0 but get {args.t}"
    assert args.alpha > 0, f"alpha must be > 0 but get {args.alpha}"
    assert args.labeled_train_size >= 1, f"labeled_train_size must be >= 1 but get {args.labeled_train_size}"
    assert args.validation_size >= 1, f"validation_size must be >= 1 but get {args.validation_size}"
    assert args.lr_warmup_step >= 0, f"lr_warmup_step must be >= 0 but get {args.lr_warmup_step}"

    if args.data_dims is not None:
        assert all(dim > 0 for dim in args.data_dims), f"data_dims must all be > 0 but get {args.data_dims}"

    assert args.num_latest_checkpoints_kept is None or args.num_latest_checkpoints_kept >= 0, \
        f"num_latest_checkpoints_kept must be None or >= 0, but get {args.num_latest_checkpoints_kept}"

    if args.checkpoint_path is not None:
        assert Path(args.checkpoint_path).is_file(), f"\"{args.checkpoint_path}\" is invalid"
    else:
        assert not bool(args.use_pretrain), f"checkpoint_path must be set if use_pretrain is True"


def get_args(parser: Optional[ArgumentParser] = None,
             return_unparsed: bool = False) -> ArgsOutType:
    if parser is None:
        parser = get_arg_parser()

    parsed_args, unparsed = parser.parse_known_args()

    # parse CLI arguments
    parsed_args = parse_args(parsed_args)

    if return_unparsed:
        return parsed_args, unparsed
    elif unparsed:
        warnings.warn(f"unexpected args {unparsed}")

    return parsed_args


def generate_logger_run_name(args: Namespace) -> str:
    # auto generate name for logger run
    augmenter_type_name_map = {
        "simple": "S",
        "fixed": "H",
    }

    estimator_name_map = {
        "default": "MixMatch",
        "ablation": "Ablation",
        "simple": "SimPLE",
    }

    logger_run_name = estimator_name_map[args.estimator]

    if args.estimator == "simple" and args.lambda_pair == 0:
        logger_run_name += " Baseline"

    if args.strong_augmenter_type in augmenter_type_name_map:
        logger_run_name += f" {augmenter_type_name_map[args.strong_augmenter_type]}"

    if args.estimator != "simple":
        if args.lambda_pair != 0:
            logger_run_name += "-pair"

    if args.use_pretrain:
        logger_run_name += "-pretrain"

    if args.debug_mode:
        logger_run_name += " Debug"

    return logger_run_name


def generate_log_path(log_dir: str, basename: str) -> str:
    log_dir_path = Path(log_dir) / f"{basename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return str(log_dir_path)


def update_args(args: Namespace, args_override: Dict[str, Any], excluded_keys: Optional[Set[str]] = None,
                inplace: bool = False) -> Namespace:
    if excluded_keys is None:
        excluded_keys = set()

    if not inplace:
        # shallow copy args
        args = Namespace(**vars(args))

    args_dict = vars(args)

    # remove excluded keys from args_override
    args_override_filtered = filter_dict(args_override, excluded_keys=excluded_keys)

    # override args
    args_dict.update(args_override_filtered)

    return args


def args_to_logger_config(args: Namespace, excluded_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    if excluded_keys is None:
        excluded_keys = LOGGER_EXCLUDED_KEYS

    return filter_dict(vars(args), excluded_keys)
