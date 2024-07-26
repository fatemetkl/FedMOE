from typing import Dict, Tuple

import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

# Note: the original module throws error, so I had to override this method
# from echotorch.data.datasets import PeriodicSignalDataset
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset  # type: ignore


class TimeSeriesPeriodicDataset(BaseDataset):
    def __init__(self, data_length: int, data_size: int) -> None:
        data = PeriodicSignalDataset(sample_len=data_length, n_samples=data_size, period=[5, 6, 12, 20])
        self.data = data.outputs[0]
        # Using teacher forcing method in training
        # Shift input elements to the left to create target
        self.targets = self.data[1:]  # type: ignore
        # Transformations are already applied
        self.transform = None
        self.target_transform = None


def load_periodic_dataloader(
    train_data_size: int,
    val_data_size: int,
    batch_size: int,
    data_length: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load Periodic Signal Dataset (training and validation set). This is used to pre-train the transformer models"""
    train_ds: BaseDataset = TimeSeriesPeriodicDataset(data_length, train_data_size)
    val_ds: BaseDataset = TimeSeriesPeriodicDataset(data_length, val_data_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(val_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
    return train_loader, validation_loader, num_examples


def get_periodic_signal_sequence(n_samples: int, data_length: int) -> torch.Tensor:
    """The concatenated data sequences are used for the main algorithm (online prediction)"""
    #  For now we assume there is only one data sequence
    periodic_ds = PeriodicSignalDataset(sample_len=data_length, n_samples=n_samples, period=[5, 6, 12, 20])
    periodic_ds_tensor = torch.cat([periodic_ds[i] for i in range(0, len(periodic_ds))]).reshape(-1)
    return periodic_ds_tensor
