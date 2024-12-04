from functools import partial

import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

# Note: the original module throws error, so I had to override this method
from fedmoe.datasets.echotorch_datasets.logistic_map import LogisticMapDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData


class TimeSeriesLogisticMap(TimeSeriesData):
    def __init__(self, total_time_steps: int) -> None:
        self.total_time_steps = total_time_steps
        self.logistic_sequence = LogisticMapDataset(sample_len=self.total_time_steps + 1, n_samples=1)
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """

        input_sequence = self.logistic_sequence.outputs[0].squeeze(1)[0:-1]
        # Each x dimension should be a sequence of size torch.Size([self.total_time_steps])
        assert input_sequence.shape == (self.total_time_steps,)

        def x_func(additional_input: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
            return additional_input

        return MultiDimensionalTimeFunctionInputGenerator([partial(x_func, input_sequence)], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        output_sequence = self.logistic_sequence.outputs[0].squeeze(1)[1:]
        assert output_sequence.shape == (self.total_time_steps,)

        def y_func(
            additional_tensor: torch.Tensor,
            input_matrix: torch.Tensor,
            time_axis: torch.Tensor,
        ) -> torch.Tensor:
            return additional_tensor

        return MultiDimensionalTargetGenerator([partial(y_func, output_sequence)], y_dim=1)
