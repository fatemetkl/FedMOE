import torch

from fedmoe.datasets.data_matrix_generator import (
    InputGenerator,
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
    TargetGenerator,
)
from fedmoe.datasets.fedmoe_datasets.brownian_motion import BrownianMotionDataset


class TimeSeriesData:
    def __init__(self, total_time_steps: int, input_gen: InputGenerator, target_gen: TargetGenerator) -> None:
        assert total_time_steps > 1, "Error, total_time_step should be positive and greater than one."
        self.total_time_steps = total_time_steps
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.input_matrix = input_gen.generate_input_tensor(self.time_axis)
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
        def func_y1(x1: torch.Tensor, x2: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            return x1 + 3 * torch.pow(x2, 3)

        def func_y2(x1: torch.Tensor, x2: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            return torch.exp(x1) + torch.sin(x2)

        return MultiDimensionalTargetGenerator([func_y1, func_y2], y_dim=2)


class TimeSeriesBrownian(TimeSeriesData):

    def __init__(
        self,
        total_time_steps: int,
        n_brownian_trajectories: int,
        mu: float,
        sigma: float,
        offset: float,
    ) -> None:
        """

        Args:
            total_time_steps (int):  length of trajectory sequences in terms of time steps.
            n_brownian_trajectories (int): number of individual trajectories which is going to be y_dim.
            mu (float): mean of the distribution to create Brownian trajectories.
            sigma (float): standard deviation of the distribution to create Brownian trajectories.
            offset (float): initial value of trajectories (X(0) = offset)
        """
        self.total_time_steps = total_time_steps
        self.n_brownian_trajectories = n_brownian_trajectories
        self.mu = mu
        self.sigma = sigma
        self.offset = offset
        input_gen: InputGenerator = self.initiate_input_generator()
        target_gen: TargetGenerator = self.initiate_target_generator()
        super().__init__(total_time_steps, input_gen, target_gen)

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # Input is time step.
        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x1], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        #  Output is a Brownian motion with 'n_brownian_trajectories' trajectories.
        brownian_matrix = self.get_brownian_sequences()
        function_list = []
        for trajectory_idx in range(self.n_brownian_trajectories):
            # The function that creates each 'yn' is simply the brownian trajectory of that dimension (n)
            # across all time steps.
            def f(x_axis: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
                return brownian_matrix[:, trajectory_idx]

            function_list.append(f)
        return MultiDimensionalTargetGenerator(function_list, y_dim=self.n_brownian_trajectories)

    def get_brownian_sequences(
        self,
    ) -> torch.Tensor:
        """
        This function creates a multi-dimensional tensor where each dimension is a Brownian trajectory.
        Each trajectory has a length of time_step. Here, we assume the same mu, sigma, and offset values are
        used for every trajectory, but the BrownianMotionDataset could be constructed with a tensor of mu, sigma,
        and offset values for each individual trajectory.

        """
        data = BrownianMotionDataset(
            time_steps=self.total_time_steps,
            n_trajectories=self.n_brownian_trajectories,
            mu=self.mu * torch.ones((self.n_brownian_trajectories)),
            sigma=self.sigma * torch.ones((self.n_brownian_trajectories)),
            offset=self.offset * torch.ones((self.n_brownian_trajectories)),
        )
        trajectories_sequences = data.outputs
        # data.outputs is a 2D torch tensor of shape (time_steps, n_brownian_trajectories)
        assert trajectories_sequences.shape == (self.total_time_steps, self.n_brownian_trajectories)
        return trajectories_sequences
