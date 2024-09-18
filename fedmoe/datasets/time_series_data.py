from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from fedmoe.datasets.data_matrix_generator import (
    InputGenerator,
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
    TargetGenerator,
)


class TimeSeriesData:

    def __init__(
        self, total_time_steps: int, input_gen: InputGenerator, target_gen: Optional[TargetGenerator] = None
    ) -> None:
        assert total_time_steps > 1, "Error, total_time_step should be positive and greater than one."
        self.total_time_steps = total_time_steps
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.input_matrix = input_gen.generate_input_tensor(self.time_axis)
        if target_gen is not None:
            self.target_matrix = target_gen.generate_target_tensor(self.time_axis, self.input_matrix)

    def visualize(
        self,
        server_prediction: List[torch.Tensor],
        plot_path: str,
        target_matrix: Optional[torch.Tensor] = None,
        T: int = 0,
    ) -> None:
        """
        Saves plots of input_matrix, target_matrix, and prediction_matrix.
        Each matrix has shape (total_time_steps, num_dimensions).
        Plots each dimension (row) as a line.

            Args:
                server_prediction (List[torch.Tensor]): List of predictions made by server
                plot_path (str): the plot path (including name and location) to save the plot
                target_matrix (Optional[torch.Tensor], optional): target matrix if generated outside dataset class.
        """

        # If the target_matrix is not generated in this class, we would still need it for visualization
        if self.target_matrix is None:
            self.target_matrix = target_matrix

        server_matrix = torch.stack(server_prediction, dim=0).squeeze(-1)
        # Server prediction matrix should have the same shape and target matrix.
        assert server_matrix.shape == (self.total_time_steps, self.target_matrix.shape[1])

        plt.figure(figsize=(10, 6))

        for i in range(self.input_matrix.shape[1]):
            plt.plot(self.time_axis, self.input_matrix[:, i], label=f"Input: x{i+1}", linestyle="--")

        for i in range(self.target_matrix.shape[1]):
            plt.plot(self.time_axis, self.target_matrix[:, i], label=f"Target: Y{i+1}", linestyle=":")

        for i in range(server_matrix.shape[1]):
            plt.plot(self.time_axis, server_matrix[:, i], label=f"Server prediction y{i+1}", linestyle="-")
            if T > 0:
                T_indices = [i * T for i in range(1, int(self.total_time_steps / T) + 1)]
                T_values = [server_matrix[j, i] for j in T_indices]
                plt.scatter(T_indices, T_values, marker="o", label=f"T y{i+1}")

        if T > 0:
            game_status = "with"
        else:
            game_status = "without"

        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title(f"Input, Target, and Predicted time-series, {game_status} game ")

        plt.legend()
        plt.savefig(plot_path)

        plt.close()


class TimeSeries2DXY(TimeSeriesData):
    def __init__(
        self,
        total_time_steps: int,
    ) -> None:
        self.input_gen: InputGenerator = self.initiate_input_generator()
        self.target_gen: TargetGenerator = self.initiate_target_generator()
        super().__init__(total_time_steps, self.input_gen, self.target_gen)

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # x1 = t, x2 = 2*t
        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        def func_x2(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis * 2

        return MultiDimensionalTimeFunctionInputGenerator([func_x1, func_x2], x_dim=2)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # y1  = x1 + 3x2^3 and y2= e^x1 + sin(x2)
        def func_y1(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x1 = input_matrix[:, 0]
            x2 = input_matrix[:, 1]
            return x1 + 3 * torch.pow(x2, 3)

        def func_y2(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x1 = input_matrix[:, 0]
            x2 = input_matrix[:, 1]
            return torch.exp(x1) + torch.sin(x2)

        return MultiDimensionalTargetGenerator([func_y1, func_y2], y_dim=2)
