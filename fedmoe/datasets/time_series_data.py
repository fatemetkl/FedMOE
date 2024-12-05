from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.data_matrix_generator import (
    InputGenerator,
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
    TargetGenerator,
)


class TimeSeriesTorchDataset(BaseDataset):
    def __init__(
        self,
        data: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = None
        self.target_transform = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.targets is not None

        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)


class TimeSeriesData:

    def __init__(self, total_time_steps: int, input_gen: InputGenerator, target_gen: TargetGenerator) -> None:
        assert total_time_steps > 1, "Error, total_time_step should be positive and greater than one."
        self.total_time_steps = total_time_steps
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.input_gen = input_gen
        self.target_gen = target_gen
        self.input_matrix = input_gen.generate_input_tensor(self.time_axis)
        self.target_matrix = target_gen.generate_target_tensor(self.time_axis, self.input_matrix)
        self.x_dim = self.input_matrix.shape[1]
        self.y_dim = self.target_matrix.shape[1]

    def get_dataloader(self, num_samples: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        This function can be used to generate data loaders for train or validation, which are mainly
        used to pre-train the transformer model.
        Set the shuffle variable to True for validation data loader.
        """
        # Generate new data samples
        data: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for sample in range(num_samples):
            sample_input = self.input_gen.generate_input_tensor(self.time_axis)
            data.append(sample_input)
            targets.append(self.target_gen.generate_target_tensor(self.time_axis, sample_input))
        # for transformer training, we are interested to predict Y_{t+1} with input x_t
        # Therefore, we should shift the target matrix by one time step to bigger ts.
        last_value = targets[-1]
        # Append the last_value to the end of the target sequence.
        # This is to make sure that the length of target sequence is equal to the
        # length of the input sequence (in terms of time steps).
        shifted_target = targets[1:]
        shifted_target.append(last_value)
        dataset: BaseDataset = TimeSeriesTorchDataset(data, shifted_target)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        # Each item (input or output) in the data_loader will have a shape of (batch_size, time_steps, dim)
        return data_loader

    def visualize_input(
        self,
        plot_path: str,
        plot_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves a plot showing the input_matrix.
            Args:
                plot_path (str): the plot path (including name and location) to save the plot
                plot_info: (Optional[Dict[str, Any]]): additional information of the experiment setting to be
                added to the plot.
        """
        for i in range(self.input_matrix.shape[1]):
            plt.plot(self.time_axis, self.input_matrix[:, i], label=f"Input: x{i+1}", linestyle="--")

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content, ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        plt.xlabel("Time Steps")
        plt.ylabel("Input Value")
        plt.title("Input")

        plt.legend()
        plt.savefig(plot_path)

        plt.close()

    def visualize_server_prediction(
        self,
        server_prediction: List[torch.Tensor],
        plot_path: str,
        game_played: bool = False,
        T: int = 0,
        show_points: Optional[bool] = False,
        plot_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves plots of target_matrix and prediction_matrix.
        Each matrix has shape (total_time_steps, num_dimensions).
        Plots each dimension (row) as a line.

            Args:
                server_prediction (List[torch.Tensor]): List of predictions made by server.
                plot_path (str): the plot path (including name and location) to save the plot.
                game_played (bool): flag indicating if the game if played or not.
                T (int): the value of synchronization frequency. If T > 0 and game_played is true,
                   plot will highlight the synchronization points.
                show_points (Optional[bool]): if True, the plot will show the synchronization points as points.
                   Otherwise, it will show as vertical lines.
                plot_info: (Optional[Dict[str, Any]]): additional information of the experiment setting to be
                    added to the plot.
        """
        if game_played:
            assert T > 0, "Error: if the game is played, T should be greater than zero."

        server_matrix = torch.stack(server_prediction, dim=0).squeeze(-1)
        assert server_matrix.shape == (self.total_time_steps, self.target_matrix.shape[1]), {
            f"Error:server output matrix has a shape {server_matrix.shape},\
                but it should be{(self.total_time_steps, self.target_matrix.shape[1])}"
        }
        # Plot target y
        # it starts from y_1 because we never predict y_0 in the algorithm.
        for i in range(self.target_matrix.shape[1]):
            # Target matrix is y(t) when generated, to get y(t+1) we need to exclude the last row.
            plt.plot(self.time_axis, self.target_matrix[:, i], label=f"Target: y{i+1}", linestyle=":")

        # Plot server's prediction
        for i in range(server_matrix.shape[1]):
            plt.plot(
                self.time_axis, server_matrix[:, i].detach().numpy(), label=f"Server prediction Y{i+1}", linestyle="-"
            )
            # Display synchronization steps as points
            if game_played and show_points:
                T_indices = [i * T for i in range(1, int(self.total_time_steps / T))]
                T_values = [server_matrix[j, i].detach().numpy() for j in T_indices]
                plt.scatter(T_indices, T_values, marker="o", label=f"T step for prediction Y{i+1}")

        # Display synchronization steps as vertical lines
        if game_played and not show_points:
            for j in range(1, int(self.total_time_steps / T)):
                label = "T time steps" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=0.5, label=label)

        if game_played:
            game_status = "with"
        else:
            game_status = "without"

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content, ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title(f"Target and Server predicted time-series, {game_status} the game ")

        plt.legend()
        plt.savefig(plot_path)

        plt.close()

    def visualize_clients_predictions(
        self,
        client_predictions: List[torch.Tensor],
        plot_path: str,
        plot_info: Dict[str, Any],
        server_prediction: Optional[List[torch.Tensor]] = None,
        show_target: bool = True,
        show_input: bool = False,
    ) -> None:
        """
        Saves plots showing the predictions made by individual clients.
        Important: make sure to include num_clients in the plot_info.
        Each matrix has shape (total_time_steps, num_dimensions).
        Plots each dimension (row) as a line.

            Args:
                client_predictions (List[torch.Tensor]): List of predictions made by clients.
                plot_path (str): the plot path (including name and location) to save the plot.
                plot_info: (Dict[str, Any]): additional information of the experiment setting to be added to the plot.
                    , including the number of clients.
                server_prediction (Optional[List[torch.Tensor]]): if it is passed, plot will also show the predictions
                    made by the server to help visualize how the individual predictions are combined for final
                    prediction.
                show_target (bool): default is True, indicating whether we want to visualize the target or not.

        """
        assert plot_info["num_clients"] is not None

        # Optional server prediction visualization
        if server_prediction is not None:
            server_matrix = torch.stack(server_prediction, dim=0).squeeze(-1)
            # Server prediction matrix should have the same shape as target matrix.
            assert server_matrix.shape == (self.total_time_steps, self.target_matrix.shape[1]), {
                f"Error:server output matrix has a shape {server_matrix.shape},\
                    but it should be{(self.total_time_steps, self.target_matrix.shape[1])}"
            }
            for i in range(server_matrix.shape[1]):
                plt.plot(
                    self.time_axis,
                    server_matrix[:, i].detach().numpy(),
                    label=f"Server prediction Y{i+1}",
                    linestyle="-",
                )

        # Shape of client prediction tensor should be time x num_clients x y_dim
        clients_pred_matrix = torch.stack(client_predictions, dim=0)
        assert clients_pred_matrix.shape == (
            self.total_time_steps,
            plot_info["num_clients"],
            self.target_matrix.shape[1],
        ), {
            f"Error: client prediction matrix shape is {clients_pred_matrix.shape},\
            but it should be {(self.total_time_steps, plot_info['num_clients'], self.target_matrix.shape[1])}"
        }
        if show_target:
            for i in range(self.target_matrix.shape[1]):
                plt.plot(self.time_axis, self.target_matrix[:, i], label=f"Target: y{i+1}", linestyle=":")
        if show_input:
            for i in range(self.input_matrix.shape[1]):
                plt.plot(self.time_axis, self.input_matrix[:, i], label=f"Input: x{i+1}", linestyle="--")

        for client in range(int(plot_info["num_clients"])):
            for dim in range(clients_pred_matrix.shape[2]):
                plt.plot(
                    self.time_axis,
                    clients_pred_matrix[:, client, dim],
                    label=f"Prediction: client {client}_Y{i+1}",
                    linestyle="dashdot",
                )

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content, ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        plt.xlabel("Time Steps")
        plt.ylabel("Prediction Value")
        plt.title("Individual client predictions")

        plt.legend()
        plt.savefig(plot_path)

        plt.close()

    def visualize_mixture_weights(
        self,
        clients_mixture_weights: List[torch.Tensor],
        plot_path: str,
        plot_info: Dict[str, Any],
        game_played: bool = False,
        T: int = 0,
        show_points: bool = False,
        show_lines: bool = False,
    ) -> None:
        """
        Saves a plot showing the mixture weights. Important: make sure to include num_clients in plot info.

            Args:
                clients_mixture_weights (List[torch.Tensor]): List of mixture weights as they evolve in the algorithm.
                plot_path (str): the plot path (including name and location) to save the plot
                plot_info: (Dict[str, Any]): additional information of the experiment setting to be added to the plot,
                    including the number of clients.
                game_played (bool): flag indicating if the game if played or not.
                T (int): the value of synchronization frequency. If T > 0 and game_played is true,
                   the plot will highlight the synchronization points.
                show_points (bool): if True, the plot will show the synchronization points as points.
                show_lines (bool): if True, the plot will show the synchronization steps as vertical lines.
        """
        assert plot_info["num_clients"] is not None
        if game_played:
            assert T > 0, "Error: if the game is played, T should be greater than zero."

        # Shape of client prediction tensor should be time x num_clients x 1
        mixture_weights = torch.stack(clients_mixture_weights, dim=0)
        assert mixture_weights.shape == (
            self.total_time_steps - 1,
            plot_info["num_clients"],
            1,
        ), f"Error: mixture_weights.shape is {mixture_weights.shape}, but should be (time_steps - 1 , num_clients, 1)"

        for client in range(int(plot_info["num_clients"])):
            plt.plot(
                self.time_axis[:-1],
                mixture_weights[:, client, 0],
                label=f"Weight: client{client}",
                linestyle="dashdot",
            )

            if game_played and show_points:
                # Highlight synchronization steps with points
                T_indices = [i * T for i in range(1, int((self.total_time_steps - 1) / T) + 1)]
                T_values = [mixture_weights[j, client] for j in T_indices]
                plt.scatter(T_indices, T_values, marker="o", label="T step")

        if game_played and show_lines:
            # Highlight synchronization time steps with vertical lines
            for j in range(1, int((self.total_time_steps - 1) / T) + 1):
                label = "T time steps" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=0.5, label=label)

        if game_played:
            game_status = "with"
        else:
            game_status = "without"

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content, ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        plt.xlabel("Time Steps")
        plt.ylabel("Mixture weight values")
        plt.title(f"Client mixture weights, {game_status} the game ")

        plt.legend()
        plt.savefig(plot_path)

        plt.close()


class TimeSeries2DXY(TimeSeriesData):
    """
    A time series dataset with 2-dimensional input and output.
    An example of TimeSeriesData class.

    Args:
        total_time_steps (int): The total number of time steps in the dataset.

    """

    def __init__(
        self,
        total_time_steps: int,
    ) -> None:
        super().__init__(total_time_steps, self.initiate_input_generator(), self.initiate_target_generator())

    def initiate_input_generator(self) -> MultiDimensionalTimeFunctionInputGenerator:
        # x1 = t, x2 = 2*t
        def func_x1(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis

        def func_x2(t_axis: torch.Tensor) -> torch.Tensor:
            return t_axis * 2

        return MultiDimensionalTimeFunctionInputGenerator([func_x1, func_x2], x_dim=2)

    def initiate_target_generator(self) -> MultiDimensionalTargetGenerator:
        # y1  = x1 + 3x2^3 and y2= e^x1 + sin(x2)
        def func_y1(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x1 = input_matrix[:, 0]
            x2 = input_matrix[:, 1]
            return x1 + 3 * torch.pow(x2, 3)

        def func_y2(input_matrix: torch.Tensor, t_axis: torch.Tensor) -> torch.Tensor:
            x1 = input_matrix[:, 0]
            x2 = input_matrix[:, 1]
            return torch.exp(x1) + torch.sin(x2)

        return MultiDimensionalTargetGenerator([func_y1, func_y2], y_dim=2)
