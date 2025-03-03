from pathlib import Path

import torch

from fedmoe.datasets.time_series_data import TimeSeries2DXY

torch.set_default_dtype(torch.float64)


def test_time_series_2d_xy(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
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
    plot_info = {
        "num_clients": 2,
        "T": 1,
        "d_z": 2,
        "alpha": 1,
        "gamma": 1,
        "sigma": 1,
    }
    two_d_data_obj.visualize_server_prediction(
        manual_server_prediction,
        f"{save_dir}/test_plot_server.png",
        game_played=True,
        T=2,
        plot_info=plot_info,
        show_lines=True,
    )
    two_d_data_obj.visualize_server_prediction(
        manual_server_prediction, f"{save_dir}/test_plot_server_2.png", T=2, show_points=True, plot_info=plot_info
    )
    two_d_data_obj.visualize_server_prediction(
        manual_server_prediction,
        f"{save_dir}/test_plot_server_3.png",
        T=2,
        game_played=True,
        show_points=True,
        plot_info=plot_info,
    )
    two_d_data_obj.visualize_input(f"{save_dir}/test_plot_input.png", plot_info=plot_info)


def test_visualize_clients_predictions(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 5
    num_clients = 2
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    manual_server_prediction = []
    for t in range(time_steps):
        manual_server_prediction.append(torch.Tensor([10 * t, 100 * t]))

    manual_clients_predictions = []
    for t in range(time_steps):
        client_predictions = []
        for client in range(num_clients):
            client_predictions.append(torch.Tensor([[50 * t], [100 * t * (client + 1)]]))
        manual_clients_predictions.append(torch.cat(client_predictions, dim=1))
    plot_info = {
        "num_clients": num_clients,
    }
    two_d_data_obj.visualize_clients_predictions(
        manual_clients_predictions, f"{save_dir}/client_pred_plot.png", plot_info=plot_info
    )


def test_visualize_mixture_weights(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 5
    num_clients = 2
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)

    manual_mixture_weights = []
    # Mixture weights are saved from time 0 to time t-1
    # This is because we predict y_t at time t-1.
    # Therefore, we have t-1 mixture weights after the total t steps.
    for t in range(time_steps - 1):
        mixture_weights = []
        for client in range(num_clients):
            mixture_weights.append(torch.Tensor([[(client + 1) * 0.5 * t]]))
        manual_mixture_weights.append(torch.cat(mixture_weights, dim=0))
    plot_info = {
        "num_clients": num_clients,
    }
    two_d_data_obj.visualize_mixture_weights(manual_mixture_weights, f"{save_dir}/mixture_weights_plot.png", plot_info)
    two_d_data_obj.visualize_mixture_weights(
        manual_mixture_weights,
        f"{save_dir}/mixture_weights_plot_2.png",
        plot_info,
        game_played=True,
        show_points=True,
        show_lines=True,
        T=2,
    )


def test_visualize_errors_histogram(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 15
    num_clients = 2
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    manual_server_prediction = []
    for t in range(time_steps):
        manual_server_prediction.append(torch.Tensor([10 * t, 100 * t]))

    plot_info = {
        "num_clients": num_clients,
    }
    two_d_data_obj.visualize_squared_error_histogram(
        manual_server_prediction, f"{save_dir}/errors_histogram.png", plot_info=plot_info
    )


def test_visualize_server_prediction_errors(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 10
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)
    assert two_d_data_obj.input_matrix.shape == (time_steps, 2)
    assert two_d_data_obj.target_matrix.shape == (time_steps, 2)
    manual_server_prediction = []
    for t in range(time_steps):
        manual_server_prediction.append(torch.Tensor([10 * t, 30 * t]))
    plot_info = {
        "num_clients": 2,
        "T": 1,
        "d_z": 2,
        "alpha": 1,
        "gamma": 1,
        "sigma": 1,
    }
    two_d_data_obj.visualize_server_squared_errors(
        manual_server_prediction,
        f"{save_dir}/test_plot_server_errors.png",
        game_played=True,
        T=2,
        plot_info=plot_info,
        show_lines=True,
    )
    two_d_data_obj.visualize_server_squared_errors(
        manual_server_prediction,
        f"{save_dir}/test_plot_server_errors_2.png",
        game_played=True,
        T=2,
        show_points=True,
        plot_info=plot_info,
    )


def test_visualize_clients_errors(tmp_path: Path) -> None:
    save_dir = tmp_path.joinpath("artifacts")
    save_dir.mkdir()
    # Set this value to true if you want to see the generated plots
    save_plots = False
    if save_plots:
        save_dir = Path("fedmoe/tests/datasets/artifacts")
    # variables
    time_steps = 10
    num_clients = 2
    two_d_data_obj = TimeSeries2DXY(total_time_steps=time_steps)

    manual_clients_predictions = []
    for t in range(time_steps):
        client_predictions = []
        for client in range(num_clients):
            client_predictions.append(torch.Tensor([[50 * t], [100 * t * (client + 1)]]))
        manual_clients_predictions.append(torch.cat(client_predictions, dim=1))
    plot_info = {
        "num_clients": num_clients,
    }
    two_d_data_obj.visualize_client_squared_errors(
        manual_clients_predictions,
        f"{save_dir}/client_error_plot.png",
        plot_info=plot_info,
        game_played=True,
        T=2,
        show_points=True,
    )

    two_d_data_obj.visualize_client_squared_errors(
        manual_clients_predictions,
        f"{save_dir}/client_error_plot_2.png",
        plot_info=plot_info,
        game_played=True,
        T=2,
        show_points=False,
        show_lines=True,
    )
