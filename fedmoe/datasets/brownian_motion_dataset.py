from functools import partial
from typing import Callable, List

import torch

from fedmoe.datasets.data_matrix_generator import (
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
)
from fedmoe.datasets.fedmoe_datasets.brownian_motion import BrownianMotionDataset
from fedmoe.datasets.time_series_data import TimeSeriesData


def get_brownian_sequences_fixed_mu_sigma(
    total_time_steps: int,
    n_brownian_trajectories: int,
    mu: float,
    sigma: float,
    offset: float,
) -> torch.Tensor:
    """
    This function creates a multi-dimensional tensor where each dimension is a Brownian trajectory.
    Each trajectory has a length of time_step. Here, we assume the same mu, sigma, and offset values are
    used for every trajectory, but the BrownianMotionDataset could be constructed with a tensor of mu, sigma,
    and offset values for each individual trajectory.
    Args:
        total_time_steps (int):  length of trajectory sequences in terms of time steps.
        n_brownian_trajectories (int): number of individual trajectories which is going to be y_dim.
        mu (float): mean of the distribution to create Brownian trajectories.
        sigma (float): standard deviation of the distribution to create Brownian trajectories.
        offset (float): initial value of trajectories (X(0) = offset)
    Returns:
        torch.Tensor: 2D torch tensor with first dimension representing time steps,
          and the second dimension representing the value of all the trajectories at that time step.
    """
    # Here we are setting the same value for mu and sigma for all the trajectories, but you could
    # specify different distribution values for each trajectory.
    # For a standard Brownian, the offset should be zero (or None).
    data = BrownianMotionDataset(
        time_steps=total_time_steps,
        n_trajectories=n_brownian_trajectories,
        mu=mu * torch.ones((n_brownian_trajectories)),
        sigma=sigma * torch.ones((n_brownian_trajectories)),
        offset=offset * torch.ones((n_brownian_trajectories)),
    )
    trajectories_sequences = data.outputs
    # data.outputs is a 2D torch tensor of shape (time_steps, n_brownian_trajectories)
    assert trajectories_sequences.shape == (total_time_steps, n_brownian_trajectories)
    return trajectories_sequences


class TimeSeriesBrownianTarget(TimeSeriesData):

    def __init__(
        self,
        total_time_steps: int,
        n_brownian_trajectories: int,
        mu: float,
        sigma: float,
        offset: float,
    ) -> None:
        """
        In this dataset, input matrix is the time axis, and output is a matrix of Brownian motion trajectories.
        In a time-series dataset, we always have time steps that start from 0 to total_time_steps. These time-steps
        are referred to as t=0 to t=total_time_steps-1. Inputs or outputs could also be multi-dimensional,
        where we index them x1 to xn, denoting dimension 1 to dimension n in input, and y1 to yn in output.
        In this dataset we have a 1D input, and nD output, where n is the number of Brownian motion trajectories.
        BM(n) refers to the nth Brownian motion trajectory.
        x = t, y = [y1 = BM(1), y2=BM(2), ...,yn = BM(n)]
        dim_x = 1, dim_y = n_brownian_trajectories

        Args:
            total_time_steps (int):  length of trajectory sequences in terms of time steps.
            n_brownian_trajectories (int): number of individual trajectories which is going to be y_dim.
            mu (float): mean of the distribution to create Brownian trajectories.
            sigma (float): standard deviation of the distribution to create Brownian trajectories.
            offset (float): the initial value of Brownian trajectories at time t=0,
                            this is where the trajectories start from.
        """
        self.total_time_steps = total_time_steps
        self.n_brownian_trajectories = n_brownian_trajectories
        self.mu = mu
        self.sigma = sigma
        self.offset = offset
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # Input is time step.

        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        return MultiDimensionalTimeFunctionInputGenerator([func_x1], x_dim=1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        #  Output is a Brownian motion with 'n_brownian_trajectories' trajectories.
        brownian_matrix = get_brownian_sequences_fixed_mu_sigma(
            self.total_time_steps, self.n_brownian_trajectories, self.mu, self.sigma, self.offset
        )
        function_list: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = []
        for trajectory_idx in range(self.n_brownian_trajectories):
            # The function that creates each 'yn' is simply the brownian trajectory of that dimension (n)
            # across all time steps.
            trajectory = brownian_matrix[:, trajectory_idx]

            def f(additional_target_tensor: torch.Tensor, input_x: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
                return additional_target_tensor

            function_list.append(partial(f, trajectory))
        return MultiDimensionalTargetGenerator(function_list, y_dim=self.n_brownian_trajectories)


class BrownianSequenceAddition(TimeSeriesData):

    def __init__(
        self,
        total_time_steps: int,
        n_brownian_trajectories: int,
        mu: float,
        sigma: float,
        offset: float,
    ) -> None:
        """
        In a time-series dataset, we always have time steps that start from 0 to total_time_steps. These time-steps
        are referred to as t=0 to t=total_time_steps-1. Inputs or outputs could also be multi-dimensional,
        where we index them x1 to xn, denoting dimension 1 to dimension n in input, and y1 to yn in output.
        In this class, input matrix is x1 appended to a Brownian motion matrix (x2,..xn rows are Brownian motion
        trajectories, and xn = BM(n)). x1 can be defined like a separate deterministic trajectory,
        in this example it is x1: [0.0, 0.1, 0.2, ....].
        Output matrix is defined such that for each output dimension yn we have: yn = BM(n) + x1, and B(n) is the nth
        trajectory in the Brownian motion.
        x = [x1, BM(1), BM(2), ... BM(n_brownian_trajectories)],
        y = [x1 + BM(1), x1 + BM(2), ..., x1+ BM(n_brownian_trajectories)]

        Args:
            total_time_steps (int):  length of trajectory sequences in terms of time steps.
            n_brownian_trajectories (int): number of individual trajectories which is going to be y_dim.
            mu (float): mean of the distribution to create Brownian trajectories.
            sigma (float): standard deviation of the distribution to create Brownian trajectories.
            offset (float): the initial value of Brownian trajectories at time t=0,
                            this is where the trajectories start from.
        """
        self.total_time_steps = total_time_steps
        self.n_brownian_trajectories = n_brownian_trajectories
        self.mu = mu
        self.sigma = sigma
        self.offset = offset
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        function_list: List[Callable[[torch.Tensor], torch.Tensor]] = []
        brownian_matrix = get_brownian_sequences_fixed_mu_sigma(
            self.total_time_steps, self.n_brownian_trajectories, self.mu, self.sigma, self.offset
        )

        # Example: x1: [0.0, 0.1, 0.2, ....]
        def x1_func(t_axis: torch.Tensor) -> torch.Tensor:
            return 0.1 * t_axis

        function_list.append(x1_func)

        for trajectory_idx in range(self.n_brownian_trajectories):
            trajectory = brownian_matrix[:, trajectory_idx]

            def f(additional_input: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
                return additional_input

            function_list.append(partial(f, trajectory))
        return MultiDimensionalTimeFunctionInputGenerator(function_list, x_dim=self.n_brownian_trajectories + 1)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        function_list: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = []
        for trajectory_idx in range(1, self.n_brownian_trajectories + 1):
            # The function that creates each 'yn' is simply the brownian trajectory of that dimension (n)
            # across all time steps added to x1 (x1: [0.0, 0.1, 0.2, ....]).
            def yn_func(idx: int, input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
                # Add the first dimension (x1) to the other ones (x2, x3, ...x{num_trajectories})
                x1 = input_matrix[:, 0]
                return input_matrix[:, idx] + x1

            function_list.append(partial(yn_func, trajectory_idx))
        return MultiDimensionalTargetGenerator(function_list, y_dim=self.n_brownian_trajectories)
