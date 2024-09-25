from pathlib import Path

import torch

from fedmoe.datasets.time_series_data import TimeSeries2DXY


def test_time_series_2d_xy(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 5
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    assert two_d_data_obj.input_matrix.shape == (time_steps, 2)
    assert two_d_data_obj.target_matrix.shape == (time_steps, 2)
    manual_server_prediction = []
    for t in range(time_steps):
        manual_server_prediction.append(torch.Tensor([10 * t, 30 * t]))
    two_d_data_obj.visualize(manual_server_prediction, f"{save_dir}/test_plot.png", T=2)
    two_d_data_obj.visualize(manual_server_prediction, f"{save_dir}/test_plot_2.png", T=2, show_points=True)
