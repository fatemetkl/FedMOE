import torch

from fedmoe.datasets.synthetic_datasets.brownian_motion import BrownianMotionDataset


def get_periodic_signal_sequence(n_samples: int, data_length: int) -> torch.Tensor:
    data = BrownianMotionDataset(sample_len=data_length, n_samples=n_samples)
    # data.outputs is a 2D torch tensor of shape (self.paths, self.points) which translates to (x_dim, time_length)
    # We need to transpose it to have the shape (time_length, x_dim)
    data_sequence = data.outputs.T
    assert data_sequence.shape == (data_length, n_samples)
    return data_sequence


def visalize_brownian_data(dataste: BrownianMotionDataset) -> None:
    dataste.visualize()
