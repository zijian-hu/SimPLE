from .cifar10_datamodule import CIFAR10DataModule
from .domainnet_real import DomainNetReal
from .utils import per_class_random_split

# for type hint
from typing import Optional, Tuple, List
from torch.utils.data import Dataset


class DomainNetRealDataModule(CIFAR10DataModule):
    num_classes: int = 345

    total_train_size: int = 120_906
    total_test_size: int = 52_041

    DATASET = DomainNetReal

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 dims: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        if dims is None:
            dims = (3, 224, 224)

        super(DomainNetRealDataModule, self).__init__(
            data_dir=data_dir,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            dims=dims,
            **kwargs)

        # dataset stats
        # DomainNet-Real mean, std values in CHW
        self.dataset_mean = [0.54873651, 0.60511086, 0.5840634]
        self.dataset_std = [0.33955591, 0.32637834, 0.31887854]

    def split_dataset(self, dataset: Dataset, **kwargs) -> List[Dataset]:
        split_kwargs = dict(lengths=[self.validation_size, self.labeled_train_size],
                            num_classes=self.num_classes,
                            uneven_split=True)

        # update split arguments
        split_kwargs.update(kwargs)

        return per_class_random_split(dataset, **split_kwargs)
