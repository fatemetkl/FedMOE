from abc import ABC, abstractmethod

import torch

from fedmoe.datasets.fedmoe_datasets.brownian_motion import BrownianMotionDataset


class TimeSeriesData(ABC):
    def __init__(self, total_time_steps: int) -> None:
        self.total_time_steps = total_time_steps
        self.time_axis = torch.range(0, self.total_time_steps - 1)
        self.input_matrix = self.generate_input_tensor()
        self.target_matrix = self.generate_target_tensor()

    @abstractmethod
    def generate_input_tensor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_target_tensor(self) -> torch.Tensor:
        """
        Implementation of how to generate output at each time step which can depend on input as well.
        Ideally, we want to have a relation between input and output sequences,
        to make sure output is relevant to input, but this is not enforced.
        """
        pass


class TimeSeriesXYSequences(TimeSeriesData):
    def __init__(
        self,
        total_time_steps: int,
    ) -> None:
        super().__init__(total_time_steps)

    def generate_input_tensor(self) -> torch.Tensor:
        # x1 = t, x2 = 2*t
        self.x1 = self.time_axis
        self.x2 = 2 * self.time_axis
        input_matrix = torch.cat([self.x1, self.x2], dim=1)
        assert input_matrix.shape == (self.total_time_steps, 2)
        return input_matrix

    def generate_target_tensor(self) -> torch.Tensor:
        # y1  = x1 + 3x2^3 and y2= e^x1 + sin(x2)
        self.y1 = self.x1 + 3 * torch.pow(self.x2, 3)
        self.y2 = torch.exp(self.x1) + torch.sin(self.x2)
        output_matrix = torch.cat([self.y1, self.y2], dim=1)
        assert output_matrix.shape == (self.total_time_steps, 2)
        return output_matrix


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
            n_brownian_trajectories (int): number of individual trajectories.
            mu (float): mean of the distribution to create Brownian trajectories.
            sigma (float): standard deviation of the distribution to create Brownian trajectories.
            offset (float): initial value of trajectories (X(0) = offset)
        """
        super().__init__(total_time_steps)
        self.n_brownian_trajectories = n_brownian_trajectories
        self.mu = mu
        self.sigma = sigma
        self.offset = offset

    def generate_input_tensor(self) -> torch.Tensor:
        # input is time step and output
        input_matrix = self.time_axis
        assert input_matrix.shape == (self.total_time_steps, 1)
        return input_matrix

    def generate_target_tensor(self) -> torch.Tensor:
        #  Output is a Brownian motion with
        return self.get_brownian_sequences()

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
