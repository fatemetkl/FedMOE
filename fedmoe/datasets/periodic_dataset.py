from functools import partial
from typing import Dict, List, Tuple

import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

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
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """
        periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps, n_samples=1, period=self.period_list
        )
        # We just take the first sequence because n_samples is one.
        input_sequence = periodic_sequence.outputs[0].squeeze(1)
        assert input_sequence.shape == (self.total_time_steps,)

        def x_func(additional_input: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
            return additional_input

        return MultiDimensionalTimeFunctionInputGenerator([partial(x_func, input_sequence)], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        def y_func(
            input_matrix: torch.Tensor,
            time_axis: torch.Tensor,
        ) -> torch.Tensor:
            last_value = input_matrix[-1]
            # Append the last_value to the end of the target sequence.
            # This is to make sure that the length of target sequence is equal to the
            # length of the input sequence (in terms of time steps).
            input_matrix = torch.cat((input_matrix.squeeze(1), last_value), dim=0)
            return input_matrix[1:]

        return MultiDimensionalTargetGenerator([y_func], y_dim=1)

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
