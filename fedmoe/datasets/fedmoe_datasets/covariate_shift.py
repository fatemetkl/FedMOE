import math

import matplotlib.pyplot as plt
import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)
from fedmoe.datasets.time_series_data import TimeSeriesData


class CovariateShiftDataset(TimeSeriesData):
    def __init__(self, total_time_steps: int) -> None:
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # Generate a uniform x_1, x_2, x_3 from 0 to 2pi and cut into total_time_steps
        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps)
        
        def func_x2(t_axis: torch.Tensor) -> torch.Tensor:
            return torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps)
        
        def func_x3(t_axis: torch.Tensor) -> torch.Tensor:            
            return torch.sqrt(torch.linspace(start=0.0, end=2 * math.pi, steps=self.total_time_steps))

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
            y_1_1 = torch.pow(x_1, 2) + torch.sin(x_2) + torch.mul(x_1, x_3) + + 0.5*torch.cos(10*x_1)
            y_1_2 = x_1 + x_2 - torch.sin(x_3)
            return torch.mul(mixing_weights, y_1_1) + torch.mul(1.0 - mixing_weights, y_1_2)

        def func_y2(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x_1 = input_matrix[:, 0]
            x_2 = input_matrix[:, 1]
            x_3 = input_matrix[:, 2]
            mixing_weights = self._get_mixing_weight(x_1)
            y_2_1 = torch.mul(x_1, torch.cos(x_2)) + x_3 - torch.exp(-x_2)
            y_2_2 = torch.mul(torch.cos(x_1), torch.sin(x_2)) + torch.pow(x_3, 2) + 0.25*torch.cos(10*x_1)
            return torch.mul(mixing_weights, y_2_1) + torch.mul(1.0 - mixing_weights, y_2_2)

        return MultiDimensionalTargetGenerator([func_y1, func_y2], y_dim=2)

    def visualize(self) -> None:
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        for target_path in range(n_targets):
            ax.plot(
                self.input_matrix[:, 0].squeeze(),
                self.target_matrix[:, target_path],
                linestyle="dotted",
                linewidth=1.5,
            )
        ax.set_title("Values of Y")
        ax.set_xlabel("x_1")
        ax.set_ylabel("Value")
        plt.show()

        mixing_weights = self._get_mixing_weight(self.input_matrix[:, 0])
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.input_matrix[:, 0].squeeze(), mixing_weights.squeeze(), linestyle="dotted", linewidth=1.5)
        ax.set_title("Values of Mixture Weight")
        ax.set_xlabel("x_1")
        ax.set_ylabel("Value")
        plt.show()
