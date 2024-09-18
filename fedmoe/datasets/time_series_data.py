from typing import Optional

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
