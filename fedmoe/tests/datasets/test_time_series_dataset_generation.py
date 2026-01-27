import math

import torch

from fedmoe.datasets.brownian_motion_dataset import BrownianSequenceAddition, TimeSeriesBrownianTarget
from fedmoe.datasets.fedmoe_datasets.concept_drift import ConceptDriftDataset
from fedmoe.datasets.logistic_map_dataset import TimeSeriesLogisticMap
from fedmoe.datasets.periodic_dataset import TimeSeriesPeriodic
from fedmoe.datasets.simple_datasets import TimeSeriesLinearLine
from fedmoe.datasets.time_series_data import TimeSeries2DXY


torch.set_default_dtype(torch.float64)


def test_linear_line() -> None:
    dataset = TimeSeriesLinearLine(10, 2, 1)
    assert dataset.input_matrix.shape == (10, 1)
    assert dataset.target_matrix.shape == (10, 1)

    # Input should be time axis (shift to 1 because we trim the first x_0)
    assert torch.allclose(dataset.input_matrix, torch.linspace(1.0, 10.0, 10).reshape(-1, 1))
    # Output should be 2*x + 1 starting at x=0 to x=9 (since x_t generates y_{t+1})
    target_matrix = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]).reshape(-1, 1)
    assert torch.allclose(dataset.target_matrix, target_matrix)


def test_concept_drift() -> None:
    dataset = ConceptDriftDataset(10)
    assert dataset.input_matrix.shape == (10, 3)
    assert dataset.target_matrix.shape == (10, 2)

    x_1_target = torch.linspace(0, 2 * math.pi, 11)[1:].reshape(-1, 1)
    x_2_target = x_1_target
    x_3_target = torch.sqrt(x_1_target)
    assert torch.allclose(torch.cat((x_1_target, x_2_target, x_3_target), dim=1), dataset.input_matrix)

    # Should correspond to x_1=x_2=x_3=0.0 as the generating input
    assert dataset.target_matrix[0, 0].item() == 0.5
    assert dataset.target_matrix[0, 1].item() == -1


def test_time_series_2d_xy() -> None:
    # variables
    time_steps = 5
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    assert two_d_data_obj.input_matrix.shape == (time_steps, 2)
    assert two_d_data_obj.target_matrix.shape == (time_steps, 2)

    manual_input_matrix = torch.Tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    assert torch.allclose(two_d_data_obj.input_matrix, manual_input_matrix, rtol=0.0, atol=1e-5)

    # Remember that x_t generates y_{t+1}
    x1 = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    x2 = torch.Tensor([0.0, 2.0, 4.0, 6.0, 8.0])
    # y1  = x1 + 3x2^3 and y2= e^x1 + sin(x2)
    y1 = x1 + 3 * torch.pow(x2, 3)
    y1_manual = torch.Tensor([0.0, 25.0, 194.0, 651.0, 1540.0])
    assert torch.allclose(y1, y1_manual, rtol=0.0, atol=1e-5)

    y2 = torch.exp(x1) + torch.sin(x2)
    manual_output_matrix = torch.stack((y1_manual.T, y2.T), dim=1)
    assert torch.allclose(two_d_data_obj.target_matrix, manual_output_matrix, rtol=0.0, atol=1e-3)

    # Test data loader creation
    train_loader = two_d_data_obj.get_dataloader(num_samples=100, batch_size=5)
    assert len(train_loader) == 20


def test_input_brownian_time_series_shape() -> None:
    # variables
    time_steps = 5
    brownian_data_obj = TimeSeriesBrownianTarget(
        total_time_steps=time_steps,
        n_brownian_trajectories=3,
        mu=0.0,
        sigma=1.0,
        offset=0.0,
    )
    # n_brownian_trajectories = y_dim
    assert brownian_data_obj.input_matrix.shape == (time_steps, 1)
    assert brownian_data_obj.target_matrix.shape == (time_steps, 3)

    assert torch.allclose(brownian_data_obj.input_matrix, torch.linspace(1.0, 5.0, 5).reshape(-1, 1))


def test_brownian_addition_time_series_data() -> None:
    # variables
    time_steps = 5
    brownian_data_obj = BrownianSequenceAddition(
        total_time_steps=time_steps,
        n_brownian_trajectories=3,
        mu=0.0,
        sigma=1.0,
        offset=0.0,
    )
    # n_brownian_trajectories = y_dim
    assert brownian_data_obj.input_matrix.shape == (time_steps, 4)
    assert brownian_data_obj.target_matrix.shape == (time_steps, 3)

    # Check the output follows the logic that we expect
    brownian_input1 = brownian_data_obj.input_matrix[:, 1]
    brownian_input2 = brownian_data_obj.input_matrix[:, 2]
    brownian_input3 = brownian_data_obj.input_matrix[:, 3]
    x1 = torch.Tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    y1 = torch.cat((torch.tensor([0.0]), brownian_input1[:-1]), dim=0) + x1
    y2 = torch.cat((torch.tensor([0.0]), brownian_input2[:-1]), dim=0) + x1
    y3 = torch.cat((torch.tensor([0.0]), brownian_input3[:-1]), dim=0) + x1
    manual_brownian_output = torch.stack([y1, y2, y3], dim=1)
    assert torch.allclose(brownian_data_obj.target_matrix, manual_brownian_output, rtol=0.0, atol=1e-4)

    # Test data loader creation
    train_loader = brownian_data_obj.get_dataloader(num_samples=50, batch_size=5)
    assert len(train_loader) == 10


def test_1d_periodic_time_series_data_loader() -> None:
    # variables
    time_steps = 5
    batch_size = 4
    periodic_data_obj = TimeSeriesPeriodic(total_time_steps=time_steps)
    assert periodic_data_obj.input_matrix.shape == (time_steps, 1)

    assert torch.allclose(periodic_data_obj.input_matrix, periodic_data_obj.target_matrix)

    # Test data loader
    train_loader = periodic_data_obj.get_dataloader(num_samples=20, batch_size=batch_size)
    for input, output in train_loader:
        assert input.shape == output.shape == (batch_size, time_steps - 1, 1)
    assert len(train_loader) == 5


def test_1d_logistic_map_time_series_data_loader() -> None:
    # variables
    time_steps = 10
    batch_size = 20
    periodic_data_obj = TimeSeriesLogisticMap(total_time_steps=time_steps)
    assert periodic_data_obj.input_matrix.shape == (time_steps, 1)

    assert torch.allclose(periodic_data_obj.input_matrix, periodic_data_obj.target_matrix)

    # Test data loader
    train_loader = periodic_data_obj.get_dataloader(num_samples=200, batch_size=batch_size)
    assert len(train_loader) == 10
