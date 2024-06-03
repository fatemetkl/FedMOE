from typing import Dict, Tuple

from fl4health.utils.dataset import BaseDataset  # type: ignore
from torch.utils.data import DataLoader

# Note: the original module throws error, so I had to override this method
# from echotorch.data.datasets import PeriodicSignalDataset
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset


class TimeSeriesPeriodicDataset(BaseDataset):
    def __init__(self, data_length: int, data_size: int) -> None:
        self.data = PeriodicSignalDataset(data_length, n_samples=data_size, period=[i for i in range(9)])
        # Using teacher forcing method in training
        # Shift input elements to the left to create target
        self.targets = self.data[1:]  # type: ignore
        # Transformations are already applied
        self.transform = None
        self.target_transform = None


def load_periodic_dataset(
    train_data_size: int,
    val_data_size: int,
    batch_size: int,
    data_length: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load Periodic Signal Dataset (training and validation set)."""
    train_ds: BaseDataset = TimeSeriesPeriodicDataset(data_length, train_data_size)
    val_ds: BaseDataset = TimeSeriesPeriodicDataset(data_length, val_data_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(val_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
    return train_loader, validation_loader, num_examples
