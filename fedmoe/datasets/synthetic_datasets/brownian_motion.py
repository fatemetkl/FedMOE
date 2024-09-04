import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class BrownianMotionDataset(Dataset):
    """
    Brownian motion dataset with configurable drift(mean) and Volatility (standard deviation).
    Source: https://www.quantstart.com/articles/brownian-motion-simulation-with-python/
    """

    # Constructor
    def __init__(
        self, sample_len: int, n_samples: int, mu: float = 0.0, sigma: float = 1.0, dtype: torch.dtype = torch.float64
    ) -> None:
        # Properties
        # Number of paths is equal to the n_samples
        # Number of the data points is equal to sample_len
        self.points = sample_len
        self.paths = n_samples
        # Normal distribution parameters
        self.mu = mu
        self.sigma = sigma
        self.dtype = dtype

        rng = np.random.default_rng(42)
        # Set the initial set of random normal draws
        self.Z = rng.normal(self.mu, self.sigma, (self.paths, self.points))

        interval = [0.0, self.points]
        self.dt = (interval[1] - interval[0]) / (self.points - 1)
        self.time_axis = torch.linspace(interval[0], interval[1], self.points)

        self.w_matrix = torch.zeros((self.paths, self.points))

        # Generate data set
        self.outputs = self._generate()

    # end __init__

    # Length
    def __len__(self) -> int:
        """
        Length
        :return:
        """
        return self.points

    # end __len__

    # Get item
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get item
        :param idx:
        :return:
        """
        return self.outputs[:, idx]

    # end __getitem__

    # Generate
    def _generate(self) -> torch.Tensor:
        """
        Generate dataset
        :return:
        """
        # List of samples
        for idx in range(self.points - 1):
            next_idx = idx + 1
            self.w_matrix[:, next_idx] = (
                self.w_matrix[:, next_idx - 1] + (self.mu * self.dt) + self.sigma * np.sqrt(self.dt) * self.Z[:, idx]
            )

        return self.w_matrix

    # end _generate

    def visualize(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for path in range(self.paths):
            ax.plot(self.time_axis, self.w_matrix[path, :])
        ax.set_title(
            f"Constant mean (mu = {self.mu}) and standard deviation (sigma = {self.sigma}) Brownian Motion paths"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.show()


# end BrownianMotionDataset
