from typing import Dict, List, Tuple

import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.data_matrix_generator import InputGenerator, MultiDimensionalTimeFunctionInputGenerator

# Note: the original module throws error, so I had to override this method
# from echotorch.data.datasets import PeriodicSignalDataset
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData


class TimeSeriesPeriodicDataset(BaseDataset):
    def __init__(self, data_length: int, data_size: int, period_list: List[int] = [5, 6, 12, 20]) -> None:
        data = PeriodicSignalDataset(sample_len=data_length, n_samples=data_size, period=period_list)
        self.data = data.outputs
        # Using teacher forcing method in training
        # Shift input elements to the left to create target
        self.targets = self.data[1:]  # type: ignore
        # Transformations are already applied
        self.transform = None
        self.target_transform = None


class TimeSeriesPeriodic(TimeSeriesData):
    def __init__(self, total_time_steps: int, period_list: List[int] = [5, 6, 12, 20]) -> None:
        self.period_list = period_list
        self.total_time_steps = total_time_steps
        # This class only generates the input matrix. The target matrix will be
        # defined by default in the client manager class, but we could also define it here.
        input_gen: InputGenerator = self.initiate_input_generator()
        super().__init__(total_time_steps, input_gen)

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """
        periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps, n_samples=1, period=self.period_list
        )
        # We just take the first sequence because n_samples is one.
        input_matrix = periodic_sequence.outputs[0]
        assert input_matrix.shape == (self.total_time_steps, 1)

        def x_func(time_step: torch.Tensor) -> torch.Tensor:
            return input_matrix[time_step]

        return MultiDimensionalTimeFunctionInputGenerator([x_func], x_dim=1)

    def load_periodic_dataloader(
        self,
        train_data_size: int,
        val_data_size: int,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Load Periodic Signal Dataset (training and validation set).
        This is used to pre-train the transformer models"""
        train_ds: BaseDataset = TimeSeriesPeriodicDataset(self.total_time_steps, train_data_size)
        val_ds: BaseDataset = TimeSeriesPeriodicDataset(self.total_time_steps, val_data_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(val_ds, batch_size=batch_size)

        num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
        return train_loader, validation_loader, num_examples
