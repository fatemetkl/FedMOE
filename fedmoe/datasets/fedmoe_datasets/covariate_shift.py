import math

import matplotlib.pyplot as plt
import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)
from fedmoe.datasets.time_series_data import TimeSeriesData

torch.set_default_dtype(torch.float64)


class CovariateShiftDataset(TimeSeriesData):
    def __init__(self, total_time_steps: int, one_dim: bool = False) -> None:
        """
        In this dataset we simulate a continuous but fast covariate shift in the relationship of x_t to y_t+1
        NOTE: By convention, at time step t, we are making predictions for y_{t+1} using x_t.
        So x_t generates y_{t+1} according to the relationship described below. See documentation in TimeSeriesData
        for more details

        x_1 and x_2 are evenly spaced from 0 to 2pi, x_3 is the square root of x_1

        In the first phase:
        y_{1, 1} = x_1^2 + sin x_2 + x_1 * x_3 + 0.5 * cos(10 * x_1)
        y_{2, 1} = x_1 * cos x_2 + x_3 - e^{-x_2}

        In the second phase:
        y_{1, 2} = x_1 + x_2 - sin x_3
        y_{2, 2} = cos x_1 * sin x_2 + x_3^2 + 0.25 * cos(10 * x_1)

        The phase transition is facilitated by the piecewise function. Note that the function is continuous. So
        the boundaries of the if conditions don't matter.
        alpha(x_1) = 1.0                    if x <= (7/8)*pi
                     cos(2(x_1-(7/8)pi))    if (7/8)*pi < x < (9/8)*pi
                     0.0                    if x >= (9/8)*pi

        such that   y_1(x_1, x_2, x_3) = alpha(x_1) * y_{1, 1} + (1-alpha(x_1)) * y_{1, 2}
                    y_2(x_1, x_2, x_3) = alpha(x_1) * y_{2, 1} + (1-alpha(x_1)) * y_{2, 2}


        Args:
            total_time_steps (int): Number of total steps to break the range of x_1, x_2, x_3 into. The larger this is
                the smaller the steps between each input value (i.e. the interval 0 to 2pi is sliced more finely)
        """
        self.one_dim = one_dim
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # Generate a uniform x_1, x_2, x_3 from 0 to 2pi and cut into total_time_steps
        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps + 1)

        def func_x2(t_axis: torch.Tensor) -> torch.Tensor:
            return torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps + 1)

        def func_x3(t_axis: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps + 1))

        return MultiDimensionalTimeFunctionInputGenerator([func_x1, func_x2, func_x3], x_dim=3)

    def _get_mixing_weight(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(x)
        weights_1 = torch.where(x <= (7 / 8) * math.pi, 1.0, 0.0)
        condition = ((7 / 8) * math.pi <= x) & (x <= (9 / 8) * math.pi)
        weights_2 = torch.where(condition, torch.cos(2 * (x - (7 / 8) * math.pi)), 0.0)
        return weights + weights_1 + weights_2

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:

        def func_y1(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x_1 = input_matrix[:, 0]
            x_2 = input_matrix[:, 1]
            x_3 = input_matrix[:, 2]

            mixing_weights = self._get_mixing_weight(x_1)
            y_1_1 = torch.pow(x_1, 2) + torch.sin(x_2) + torch.mul(x_1, x_3) + 0.5 * torch.cos(10 * x_1)
            y_1_2 = x_1 + x_2 - torch.sin(x_3)
            return torch.mul(mixing_weights, y_1_1) + torch.mul(1.0 - mixing_weights, y_1_2)

        def func_y2(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x_1 = input_matrix[:, 0]
            x_2 = input_matrix[:, 1]
            x_3 = input_matrix[:, 2]
            mixing_weights = self._get_mixing_weight(x_1)
            y_2_1 = torch.mul(x_1, torch.cos(x_2)) + x_3 - torch.exp(-x_2)
            y_2_2 = torch.mul(torch.cos(x_1), torch.sin(x_2)) + torch.pow(x_3, 2) + 0.25 * torch.cos(10 * x_1)
            return torch.mul(mixing_weights, y_2_1) + torch.mul(1.0 - mixing_weights, y_2_2)

        if not self.one_dim:
            return MultiDimensionalTargetGenerator([func_y1, func_y2], y_dim=2)
        else:
            return MultiDimensionalTargetGenerator([func_y2], y_dim=1)

    def visualize(self) -> None:
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(20, 8))
        for target_path in range(n_targets):
            ax.plot(
                self.input_matrix[:, 0].squeeze(),
                self.target_matrix[:, target_path],
                label=f"$y_{target_path+1}$",
                linestyle="-",
                linewidth=2.5,
            )

        title_font = {"family": "helvetica", "weight": "bold", "size": 35}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 35}
        plt.xticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.xlabel("$x_1$", fontdict=axis_font)
        plt.ylabel("Time-Series Values", fontdict=axis_font)
        plt.title("Concept Drift of $\\mathbf{{y}}$", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 30}, loc="upper left", labelspacing=0)
        plt.tight_layout(pad=0.5)

        plt.show()
        plt.savefig("temp.png")

        mixing_weights = self._get_mixing_weight(self.input_matrix[:, 0])
        _, ax = plt.subplots(1, 1, figsize=(24, 8))
        ax.plot(self.input_matrix[:, 0].squeeze(), mixing_weights.squeeze(), linestyle="dotted", linewidth=1.5)
        ax.set_title("Values of Mixture Weight")
        ax.set_xlabel("x_1")
        ax.set_ylabel("Value")
        plt.show()
