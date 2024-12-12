from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

torch.set_default_dtype(torch.float64)


class BrownianMotionDataset(Dataset):
    """
    Brownian motion dataset with configurable drift(mean) and Volatility (standard deviation).
    Sources: https://www.quantstart.com/articles/brownian-motion-simulation-with-python/
    https://www.bauer.uh.edu/spirrong/Monte_Carlo_Methods_In_Financial_Enginee.pdf

    """

    def __init__(
        self,
        time_steps: int,
        n_trajectories: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        random_generator_seed: int = 42,
        dtype: torch.dtype = torch.float64,
        offset: Optional[torch.Tensor] = None,
    ) -> None:
        self.time_steps = time_steps
        self.n_trajectories = n_trajectories
        # Normal distributions parameters
        assert (
            mu.shape == sigma.shape == (n_trajectories,)
        ), "You should have equal number of trajectories as the number of distribution parameters mu and sigma"
        self.mu = mu
        self.sigma = sigma
        self.dtype = dtype

        rng = np.random.default_rng(random_generator_seed)
        # Set the initial set of random standard normal draws.
        # It should be drawn from a standard distribution.
        self.Z = rng.standard_normal((self.time_steps, self.n_trajectories))

        # We assume dt (length of each time step) is 1.
        self.dt = 1
        self.time_axis = torch.arange(0, self.time_steps)

        self.w_matrix = torch.zeros((self.time_steps, self.n_trajectories))
        if offset is not None:
            assert offset.shape == (
                n_trajectories,
            ), "If specifying an offset value, you should provide one for each trajectory"
            # Setting the offset as the start value at time step zero.
            self.w_matrix[0, :] = offset

        # Generate data set
        self.outputs = self._generate()

    def __getitem__(self, time_step: int) -> torch.Tensor:
        # Returns the column of all the trajectory values at the specified time step.
        return self.outputs[time_step, :]

    def _generate(self) -> torch.Tensor:
        for time_step in range(1, self.time_steps):
            self.w_matrix[time_step, :] = (
                self.w_matrix[time_step - 1, :]
                + (self.mu * self.dt)
                + self.sigma * np.sqrt(self.dt) * self.Z[time_step - 1, :]
            )

        return self.w_matrix

    def visualize(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for path in range(self.n_trajectories):
            ax.plot(self.time_axis, self.w_matrix[:, path])
        ax.set_title("Constant mean and standard deviation Brownian Motion paths")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.show()
