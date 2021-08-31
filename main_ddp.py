import torch
from torch import distributed

import warnings

from main import main as main_single_thread
from utils import get_args, timing, set_random_seed

# for type hint
from typing import Optional
from argparse import Namespace
from utils.dataset import SSLDataModule

IS_DISTRIBUTED_AVAILABLE = distributed.is_available()


@timing
def main(args: Namespace, datamodule: Optional[SSLDataModule] = None):
    if IS_DISTRIBUTED_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        distributed.init_process_group(backend='nccl')

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        warnings.warn("Cannot initializePyTorch distributed training, fallback to single GPU training")
        device = None

    main_single_thread(args=args, datamodule=datamodule, device=device)


if __name__ == '__main__':
    parsed_args = get_args()

    # fix random seed
    set_random_seed(parsed_args.seed, is_cudnn_deterministic=parsed_args.debug_mode)

    main(parsed_args)
