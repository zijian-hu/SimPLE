from torchvision.datasets import CIFAR10

from .ssl_datamodule import SSLDataModule
from .datasets import LabeledDataset
from .utils import per_class_random_split

# for type hint
from typing import Optional, Tuple, List
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class CIFAR10DataModule(SSLDataModule):
    num_classes: int = 10

    total_train_size: int = 50_000
    total_test_size: int = 10_000

    DATASET = CIFAR10

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 dims: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        if dims is None:
            dims = (3, 32, 32)

        super(CIFAR10DataModule, self).__init__(dims=dims, **kwargs)

        self.data_dir = data_dir

        self.labeled_train_size = labeled_train_size
        self.validation_size = validation_size
        if unlabeled_train_size is None:
            self.unlabeled_train_size = self.total_train_size - self.validation_size - self.labeled_train_size
        else:
            self.unlabeled_train_size = unlabeled_train_size

        # dataset stats
        # CIFAR-10 mean, std values in CHW
        self.dataset_mean = [0.44653091, 0.49139968, 0.48215841]
        self.dataset_std = [0.26158784, 0.24703223, 0.24348513]

    def prepare_data(self, *args, **kwargs):
        self.DATASET(root=self.data_dir, train=True, download=True)
        self.DATASET(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        full_train_set = self.DATASET(root=self.data_dir, train=True)
        full_test_set = self.DATASET(root=self.data_dir, train=False)

        self.setup_helper(full_train_set=full_train_set, full_test_set=full_test_set, stage=stage)

    def setup_helper(self, full_train_set: VisionDataset, full_test_set: VisionDataset, stage: Optional[str] = None):
        self.test_set = LabeledDataset(
            full_test_set,
            root=full_test_set.root,
            min_size=self.test_min_size,
            transform=self.test_transform)

        # get subsets
        validation_subset, labeled_train_subset, unlabeled_train_subset = self.split_dataset(full_train_set)

        # convert to dataset
        self.validation_set = LabeledDataset(
            validation_subset,
            root=full_train_set.root,
            min_size=self.test_min_size,
            transform=self.val_transform)
        self.labeled_train_set = LabeledDataset(
            labeled_train_subset,
            root=full_train_set.root,
            min_size=self.train_min_size,
            transform=self.train_transform)
        self.unlabeled_train_set = LabeledDataset(
            unlabeled_train_subset,
            root=full_train_set.root,
            min_size=self.unlabeled_train_min_size,
            transform=self.train_transform)

        assert len(self.validation_set.dataset) == len(validation_subset) == self.validation_size
        assert len(self.labeled_train_set.dataset) == len(labeled_train_subset) == self.labeled_train_size
        assert len(self.unlabeled_train_set.dataset) == len(unlabeled_train_subset) == self.unlabeled_train_size
        assert len(self.test_set.dataset) == self.total_test_size

    def split_dataset(self, dataset: Dataset, **kwargs) -> List[Dataset]:
        split_kwargs = dict(lengths=[self.validation_size, self.labeled_train_size, self.unlabeled_train_size],
                            num_classes=self.num_classes,
                            uneven_split=False)

        # update split arguments
        split_kwargs.update(kwargs)

        return per_class_random_split(dataset, **split_kwargs)
