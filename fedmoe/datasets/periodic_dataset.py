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
        self.periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps + 1, n_samples=1, period=self.period_list
        )
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """
        # We just take the first sequence because n_samples is one.
        input_sequence = self.periodic_sequence.outputs[0].squeeze(1)[0:-1]
        assert input_sequence.shape == (self.total_time_steps,)

        def x_func(additional_input: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
            return additional_input

        return MultiDimensionalTimeFunctionInputGenerator([partial(x_func, input_sequence)], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        output_sequence = self.periodic_sequence.outputs[0].squeeze(1)[1:]
        assert output_sequence.shape == (self.total_time_steps,)

        def y_func(
            additional_tensor: torch.Tensor,
            input_matrix: torch.Tensor,
            time_axis: torch.Tensor,
        ) -> torch.Tensor:
            return additional_tensor

        return MultiDimensionalTargetGenerator([partial(y_func, output_sequence)], y_dim=1)


class TimeInputPeriodic(TimeSeriesData):
    def __init__(self, total_time_steps: int, period_list: List[int] = [5, 6, 12, 20]) -> None:
        self.period_list = period_list
        self.total_time_steps = total_time_steps
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """

        def func_x(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps, n_samples=1, period=self.period_list
        )
        # We just take the first sequence because n_samples is one.
        periodic_sequence_tensor = periodic_sequence.outputs[0].squeeze(1)
        assert periodic_sequence_tensor.shape == (self.total_time_steps,)

        def y_func(
            input_matrix: torch.Tensor,
            time_axis: torch.Tensor,
        ) -> torch.Tensor:
            return periodic_sequence_tensor

        return MultiDimensionalTargetGenerator([y_func], y_dim=1)
