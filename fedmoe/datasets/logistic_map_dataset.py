from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)

# Note: the original module throws error, so I had to override this method
from fedmoe.datasets.echotorch_datasets.logistic_map import LogisticMapDataset  # type: ignore
from fedmoe.datasets.time_series_data import TimeSeriesData

sns.set_style("whitegrid")
torch.set_default_dtype(torch.float64)


class TimeSeriesLogisticMap(TimeSeriesData):
    def __init__(self, total_time_steps: int) -> None:
        self.total_time_steps = total_time_steps
        # We still generate total_time_steps+1 datapoints to make the super class happy, but we'll just take the
        # first total_time_step values here
        self.logistic_sequence = LogisticMapDataset(sample_len=self.total_time_steps + 1, n_samples=1)
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def _post_process_data_matrices(
        self, input_matrix: torch.Tensor, target_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Overriding the post processing behavior to just take the first total_time_step values
        return input_matrix[:-1], target_matrix[:-1]

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        """
        This function defines how input data should be generated using PeriodicSignalDataset.
        """

        input_sequence = self.logistic_sequence.outputs[0].squeeze(1)
        # Each x dimension should be a sequence of size torch.Size([self.total_time_steps])
        assert input_sequence.shape == (self.total_time_steps + 1,)

        def x_func(additional_input: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
            return additional_input

        return MultiDimensionalTimeFunctionInputGenerator([partial(x_func, input_sequence)], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # Since x_t generates y_{t+1} the target tensor is the same as the input tensor, producing lagged inputs
        # y_0   y_1     y_2     y_3     y_4     y_5
        # y_0   y_1     y_2     y_3     y_4     y_5
        # y_0 "generates" y_1 etc.
        output_sequence = self.logistic_sequence.outputs[0].squeeze(1)
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
        plt.ylabel("Logistic Map Value", fontdict=axis_font)
        plt.title("Logistic Map Timeseries", fontdict=title_font)

        plt.tight_layout(pad=0.5)
        plt.savefig("logistic_map_dataset.pdf", format="pdf")
        plt.show()
