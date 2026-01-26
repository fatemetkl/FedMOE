from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

# Note: the original module throws error, so I had to override this method
# from echotorch.data.datasets import PeriodicSignalDataset
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData


torch.set_default_dtype(torch.float64)
sns.set_style("whitegrid")


class TimeSeriesPeriodic(TimeSeriesData):
    def __init__(self, total_time_steps: int, period_list: list[int] | None = None) -> None:
        self.period_list = period_list if period_list is not None else [5, 6, 12, 20]
        self.total_time_steps = total_time_steps
        # We still generate total_time_steps+1 datapoints to make the super class happy, but we'll just take the
        # first total_time_step values here
        self.periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps + 1,
            n_samples=1,
            period=self.period_list,
            dtype=torch.float64,
        )
        super().__init__(
            total_time_steps,
            self.initiate_input_generator(),
            self.initiate_target_generator(),
        )

    def _post_process_data_matrices(
        self, input_matrix: torch.Tensor, target_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Overriding the post processing behavior to just take the first total_time_step values
        return input_matrix[:-1], target_matrix[:-1]

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """This function defines how input data should be generated using PeriodicSignalDataset."""
        # We just take the first sequence because n_samples is one.
        input_sequence = self.periodic_sequence.outputs[0].squeeze(1)
        assert input_sequence.shape == (self.total_time_steps + 1,)

        def x_func(additional_input: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
            return additional_input

        return MultiDimensionalTimeFunctionInputGenerator([partial(x_func, input_sequence)], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Since x_t generates y_{t+1} the target tensor is the same as the input tensor, producing lagged inputs
        # y_0   y_1     y_2     y_3     y_4     y_5
        # y_0   y_1     y_2     y_3     y_4     y_5
        # y_0 "generates" y_1 etc.
        output_sequence = self.periodic_sequence.outputs[0].squeeze(1)
        assert output_sequence.shape == (self.total_time_steps + 1,)

        def y_func(
            additional_tensor: torch.Tensor,
            input_matrix: torch.Tensor,
            time_axis: torch.Tensor,
        ) -> torch.Tensor:
            return additional_tensor

        return MultiDimensionalTargetGenerator([partial(y_func, output_sequence)], y_dim=1)

    def visualize(self) -> None:
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(20, 8))
        for target_path in range(n_targets):
            sns.lineplot(x=self.time_axis, y=self.target_matrix[:, target_path], ax=ax, linestyle="solid", linewidth=3)

        title_font = {"family": "helvetica", "weight": "bold", "size": 35}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 35}
        plt.xticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Periodic Value", fontdict=axis_font)
        plt.title("Periodic Timeseries", fontdict=title_font)

        plt.tight_layout(pad=0.5)
        plt.savefig("periodic_series.pdf", format="pdf")
        plt.show()


class TimeInputPeriodic(TimeSeriesData):
    def __init__(self, total_time_steps: int, period_list: list[int] | None = None) -> None:
        self.period_list = period_list if period_list is not None else [5, 6, 12, 20]
        self.total_time_steps = total_time_steps
        super().__init__(
            total_time_steps,
            self.initiate_input_generator(),
            self.initiate_target_generator(),
        )

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """This function defines how input data should be generated using PeriodicSignalDataset."""

        def func_x(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Since x_t generates y_{t+1} and x_t is just the time index
        #  -    y_0     y_1     y_2     y_3     y_4     y_5
        # x_0   x_1     x_2     x_3     x_4     x_5      -
        periodic_sequence = PeriodicSignalDataset(
            sample_len=self.total_time_steps + 1, n_samples=1, period=self.period_list
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
