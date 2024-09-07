import torch

from fedmoe.datasets.synthetic_datasets.brownian_motion import BrownianMotionDataset


def get_brownian_data_sequences(
    n_brownian_trajectories: int, time_steps: int, mu: float, sigma: float
) -> torch.Tensor:
    """
    This function creates a multi-dimensional tensor where each dimension is a Brownian trajectory.
    Each trajectory has a length of time_step.

    Args:
        n_brownian_trajectories (int): number of individual trajectories.
        time_steps (int): length of trajectory sequences in terms of time steps.

    Returns:
        torch.Tensor: 2D torch tensor with first dimension representing time steps,
          and the second dimension representing the value of all the trajectories at that time step.
    """
    # Here we are setting the same value for mu and sigma for all the trajectories, but you could
    # specify different distribution values for each trajectory.
    data = BrownianMotionDataset(
        time_steps=time_steps,
        n_trajectories=n_brownian_trajectories,
        mu=mu * torch.ones((n_brownian_trajectories)),
        sigma=sigma * torch.ones((n_brownian_trajectories)),
    )
    data_sequence = data.outputs
    # data.outputs is a 2D torch tensor of shape (time_steps, n_brownian_trajectories)
    assert data_sequence.shape == (time_steps, n_brownian_trajectories)
    return data_sequence


def visualize_brownian_data(dataset: BrownianMotionDataset) -> None:
    dataset.visualize()
