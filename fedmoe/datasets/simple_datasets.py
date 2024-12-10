from functools import partial

import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)
from fedmoe.datasets.time_series_data import TimeSeriesData

torch.set_default_dtype(torch.float64)


class TimeSeriesLinearLine(TimeSeriesData):
    """
    x = t
    y = a * x + b
    """

    def __init__(
        self,
        total_time_steps: int,
        a: float = 2.0,
        b: float = 1.0,
    ) -> None:
        self.a = a
        self.b = b
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # x = t
        def func_x(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # y  = ax + b
        def func_y(a: float, b: float, input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            # Notice that any operation should be done on individual dimensions (xn = input_matrix[:, n-1])
            return a * input_matrix[:, 0] + b

        return MultiDimensionalTargetGenerator([partial(func_y, self.a, self.b)], y_dim=1)


class TimeSeriesQuadratic(TimeSeriesData):
    """
    x = t
    y = a * x^2 + b * x + c
    """

    def __init__(self, total_time_steps: int, a: float = 2.0, b: float = -1.0, c: float = 1.0) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # x = t
        def func_x(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # y  = a * x^2 + b * x + c
        def func_y(a: float, b: float, c: float, input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            # Notice that any operation should be done on individual dimensions (xn = input_matrix[:, n-1])
            return a * torch.pow(input_matrix[:, 0], 2) + b * input_matrix[:, 0] + c

        return MultiDimensionalTargetGenerator([partial(func_y, self.a, self.b, self.c)], y_dim=1)


class TimeSeriesSineSignal(TimeSeriesData):
    """
    x = t
    y = sin(t)
    """

    def __init__(
        self,
        total_time_steps: int,
    ) -> None:
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # x = t
        def func_x(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # y  = sin(x)
        def func_y(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            # Notice that any operation should be done on individual dimensions (xn = input_matrix[:, n-1])
            return torch.sin(input_matrix[:, 0])

        return MultiDimensionalTargetGenerator([func_y], y_dim=1)
