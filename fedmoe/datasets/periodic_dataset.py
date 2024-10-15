from functools import partial
from typing import List

import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

# Note: the original module throws error, so I had to override this method
# from echotorch.data.datasets import PeriodicSignalDataset
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData


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
