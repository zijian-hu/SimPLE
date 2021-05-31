from torchvision import transforms

# for type hint
from argparse import Namespace
from .ssl_datamodule import SSLDataModule


def get_dataset(args: Namespace, return_unlabeled_target: bool = True) -> SSLDataModule:
    if args.dataset == "cifar10":
        from .cifar10_datamodule import CIFAR10DataModule

        return CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            labeled_train_size=args.labeled_train_size,
            validation_size=args.validation_size,
            unlabeled_train_batch_size=args.unlabeled_train_batch_size,
            train_transforms=transforms.Compose([transforms.ToTensor()]),
            val_transforms=transforms.Compose([transforms.ToTensor()]),
            test_transforms=transforms.Compose([transforms.ToTensor()]))

    elif args.dataset == "cifar100":
        from .cifar100_datamodule import CIFAR100DataModule

        return CIFAR100DataModule(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            labeled_train_size=args.labeled_train_size,
            validation_size=args.validation_size,
            unlabeled_train_batch_size=args.unlabeled_train_batch_size,
            train_transforms=transforms.Compose([transforms.ToTensor()]),
            val_transforms=transforms.Compose([transforms.ToTensor()]),
            test_transforms=transforms.Compose([transforms.ToTensor()]))

    elif args.dataset == "svhn":
        from .svhn_datamodule import SVHNDataModule

        return SVHNDataModule(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            labeled_train_size=args.labeled_train_size,
            validation_size=args.validation_size,
            unlabeled_train_batch_size=args.unlabeled_train_batch_size,
            train_transforms=transforms.Compose([transforms.ToTensor()]),
            val_transforms=transforms.Compose([transforms.ToTensor()]),
            test_transforms=transforms.Compose([transforms.ToTensor()]))

    elif args.dataset == "miniimagenet":
        from .miniimagenet_datamodule import MiniImageNetDataModule

        return MiniImageNetDataModule(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            labeled_train_size=args.labeled_train_size,
            validation_size=args.validation_size,
            unlabeled_train_batch_size=args.unlabeled_train_batch_size,
            train_transforms=transforms.Compose([transforms.ToTensor()]),
            val_transforms=transforms.Compose([transforms.ToTensor()]),
            test_transforms=transforms.Compose([transforms.ToTensor()]))

    elif args.dataset == "domainnet":
        from .domainnet_datamodule import DomainNetDataModule

        return DomainNetDataModule(
            data_dir=args.data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            labeled_train_size=args.labeled_train_size,
            validation_size=args.validation_size,
            unlabeled_train_batch_size=args.unlabeled_train_batch_size,
            train_transforms=transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]),
            val_transforms=transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]),
            test_transforms=transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]))

    else:
        raise RuntimeError(f"dataset \"{args.dataset}\" is not supported")
