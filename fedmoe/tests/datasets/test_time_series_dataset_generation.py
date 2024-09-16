import torch

from fedmoe.datasets.time_series_data import TimeSeries2DXY, TimeSeriesBrownian


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


def test_brownian_time_series_shape() -> None:
    # variables
    time_steps = 5
    brownian_data_obj = TimeSeriesBrownian(
        total_time_steps=time_steps, n_brownian_trajectories=3, mu=0.0, sigma=1.0, offset=0.0
    )
    # n_brownian_trajectories = y_dim
    assert brownian_data_obj.input_matrix.shape == (time_steps, 1)
    assert brownian_data_obj.target_matrix.shape == (time_steps, 3)
