import torch

from fedmoe.datasets.synthetic_datasets.brownian_motion import BrownianMotionDataset


def get_brownian_data_sequences(
    n_brownian_trajectories: int,
    time_steps: int,
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
        n_brownian_trajectories (int): number of individual trajectories.
        time_steps (int): length of trajectory sequences in terms of time steps.
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
        time_steps=time_steps,
        n_trajectories=n_brownian_trajectories,
        mu=mu * torch.ones((n_brownian_trajectories)),
        sigma=sigma * torch.ones((n_brownian_trajectories)),
        offset=offset * torch.ones((n_brownian_trajectories)),
    )
    data_sequence = data.outputs
    # data.outputs is a 2D torch tensor of shape (time_steps, n_brownian_trajectories)
    assert data_sequence.shape == (time_steps, n_brownian_trajectories)
    return data_sequence


def visualize_brownian_data(dataset: BrownianMotionDataset) -> None:
    dataset.visualize()
