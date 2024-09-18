import torch

from fedmoe.datasets.brownian_motion_dataset import BrownianSequenceAddition, TimeSeriesBrownianTarget
from fedmoe.datasets.logistic_map_dataset import TimeSeriesLogisticMap
from fedmoe.datasets.periodic_dataset import TimeSeriesPeriodic
from fedmoe.datasets.time_series_data import TimeSeries2DXY


def test_time_series_2d_xy() -> None:
    # variables
    time_steps = 5
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    assert two_d_data_obj.input_matrix.shape == (time_steps, 2)
    assert two_d_data_obj.target_matrix.shape == (time_steps, 2)

    manual_input_matrix = torch.Tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    assert torch.allclose(two_d_data_obj.input_matrix, manual_input_matrix.double(), rtol=0.0, atol=1e-5)

    x1 = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0]).double()
    x2 = torch.Tensor([0.0, 2.0, 4.0, 6.0, 8.0]).double()
    # y1  = x1 + 3x2^3 and y2= e^x1 + sin(x2)
    y1 = x1 + 3 * torch.pow(x2, 3)
    y1_manual = torch.Tensor([0.0, 25.0, 194.0, 651.0, 1540.0]).double()
    assert torch.allclose(y1, y1_manual, rtol=0.0, atol=1e-5)

    y2 = torch.exp(x1) + torch.sin(x2)
    manual_output_matrix = torch.Tensor(
        [
            [y1_manual[0], y2[0]],
            [y1_manual[1], y2[1]],
            [y1_manual[2], y2[2]],
            [y1_manual[3], y2[3]],
            [y1_manual[4], y2[4]],
        ]
    ).double()
    assert torch.allclose(two_d_data_obj.target_matrix, manual_output_matrix, rtol=0.0, atol=1e-3)


def test_input_brownian_time_series_shape() -> None:
    # variables
    time_steps = 5
    brownian_data_obj = TimeSeriesBrownianTarget(
        total_time_steps=time_steps, n_brownian_trajectories=3, mu=0.0, sigma=1.0, offset=0.0
    )
    # n_brownian_trajectories = y_dim
    assert brownian_data_obj.input_matrix.shape == (time_steps, 1)
    assert brownian_data_obj.target_matrix.shape == (time_steps, 3)


def test_brownian_time_series_data2_shape() -> None:
    # variables
    time_steps = 5
    brownian_data_obj = BrownianSequenceAddition(
        total_time_steps=time_steps, n_brownian_trajectories=3, mu=0.0, sigma=1.0, offset=0.0
    )
    # n_brownian_trajectories = y_dim
    assert brownian_data_obj.input_matrix.shape == (time_steps, 4)
    assert brownian_data_obj.target_matrix.shape == (time_steps, 3)

    # Check the output follows the logic that we expect
    brownian_input1 = brownian_data_obj.input_matrix[:, 1]
    brownian_input2 = brownian_data_obj.input_matrix[:, 2]
    brownian_input3 = brownian_data_obj.input_matrix[:, 3]
    x1 = torch.Tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    y1 = brownian_input1 + x1
    y2 = brownian_input2 + x1
    y3 = brownian_input3 + x1
    manual_brownian_output = torch.stack([y1, y2, y3], dim=1).double()
    assert torch.allclose(brownian_data_obj.target_matrix, manual_brownian_output, rtol=0.0, atol=1e-4)


def test_1d_periodic_time_series() -> None:
    # variables
    time_steps = 5
    periodic_data_obj = TimeSeriesPeriodic(total_time_steps=time_steps)
    assert periodic_data_obj.input_matrix.shape == (time_steps, 1)

    # Test data loader
    train_loader, validation_loader, num_examples = periodic_data_obj.load_periodic_dataloader(
        train_data_size=100, val_data_size=10, batch_size=5
    )
    assert len(train_loader) == 20
    assert len(validation_loader) == 2


def test_1d_logistic_map_time_series() -> None:
    # variables
    time_steps = 10
    periodic_data_obj = TimeSeriesLogisticMap(total_time_steps=time_steps)
    assert periodic_data_obj.input_matrix.shape == (time_steps, 1)

    # Test data loader
    train_loader, validation_loader, num_examples = periodic_data_obj.load_logistic_map_dataloader(
        train_data_size=200, val_data_size=20, batch_size=4
    )
    assert len(train_loader) == 50
    assert len(validation_loader) == 5
