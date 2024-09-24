from functools import partial
from typing import Dict, Tuple

import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

# Note: the original module throws error, so I had to override this method
from fedmoe.datasets.echotorch_datasets.logistic_map import LogisticMapDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData


class LogisticMapData(BaseDataset):
    def __init__(self, data_length: int, data_size: int) -> None:
        dataset = LogisticMapDataset(sample_len=data_length, n_samples=data_size, dtype=torch.double)
        # Transformations are already applied
        self.data = dataset.outputs
        self.targets = dataset.outputs[1:]  # type: ignore
        self.transform = None
        self.target_transform = None


class TimeSeriesLogisticMap(TimeSeriesData):
    def __init__(self, total_time_steps: int) -> None:
        self.total_time_steps = total_time_steps
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """
        periodic_sequence = LogisticMapDataset(sample_len=self.total_time_steps, n_samples=1)
        input_sequence = periodic_sequence.outputs[0].squeeze(1)
        # Each x dimension should be a sequence of size torch.Size([self.total_time_steps])
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
            # The last element in tes target sequence will be the to the previous target value.
            # This is done by repeating the last element.
            last_value = input_matrix[-1]
            # Add the last_value to the end of the input_matrix
            input_matrix = torch.cat((input_matrix.squeeze(1), last_value), dim=0)
            return input_matrix[1:]

        return MultiDimensionalTargetGenerator([y_func], y_dim=1)

    def load_logistic_map_dataloader(
        self,
        train_data_size: int,
        val_data_size: int,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Load Periodic Signal Dataset (training and validation set).
        This is used to pre-train the transformer models"""
        train_ds: BaseDataset = LogisticMapData(self.total_time_steps, train_data_size)
        val_ds: BaseDataset = LogisticMapData(self.total_time_steps, val_data_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(val_ds, batch_size=batch_size)

        num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
        return train_loader, validation_loader, num_examples
