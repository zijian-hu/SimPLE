import torch
from torch import distributed

from main import main as main_single_thread
from utils import get_args, timing, set_random_seed

# for type hint
from typing import Optional
from argparse import Namespace
from utils.dataset import SSLDataModule


@timing
def main(args: Namespace, datamodule: Optional[SSLDataModule] = None):
    distributed.init_process_group(backend='nccl')

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    main_single_thread(args=args, datamodule=datamodule, device=device)


if __name__ == '__main__':
    parsed_args = get_args()

    # fix random seed
    set_random_seed(parsed_args.seed, is_cudnn_deterministic=parsed_args.debug_mode)

    main(parsed_args)
