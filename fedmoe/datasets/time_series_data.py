from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from fl4health.utils.dataset import BaseDataset
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from fedmoe.datasets.data_matrix_generator import (
    InputGenerator,
    MultiDimensionalTargetGenerator,
    MultiDimensionalTimeFunctionInputGenerator,
    TargetGenerator,
)

torch.set_default_dtype(torch.float64)


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
        self.time_axis = torch.arange(0, self.total_time_steps, dtype=torch.float64)
        self.input_gen = input_gen
        self.target_gen = target_gen
        # Since x_t generates y_{t+1}, we generate for an extra time step and trim the first x and the last y
        #  -    y_1     y_2     y_3     y_4     y_5
        # x_0   x_1     x_2     x_3     x_4
        generation_steps = torch.arange(0, self.total_time_steps + 1, dtype=torch.float64)
        self.input_matrix = input_gen.generate_input_tensor(generation_steps)
        self.target_matrix = target_gen.generate_target_tensor(generation_steps, self.input_matrix)

        self.input_matrix, self.target_matrix = self._post_process_data_matrices(self.input_matrix, self.target_matrix)
        self.x_dim = self.input_matrix.shape[1]
        self.y_dim = self.target_matrix.shape[1]

    def _post_process_data_matrices(
        self, input_matrix: torch.Tensor, target_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # trim first x and last y
        return input_matrix[1:, :], target_matrix[:-1, :]

    def get_dataloader(self, num_samples: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        This function can be used to generate data loaders for train or validation, which are mainly
        used to pre-train the transformer model.
        Set the shuffle variable to True for validation data loader.
        """
        # Generate new data samples
        data: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        # Since x_t generates y_{t+1}, we generate for an extra time step and trim the first x and the last y
        #  -    y_1     y_2     y_3     y_4     y_5
        # x_0   x_1     x_2     x_3     x_4
        generation_steps = torch.arange(0, self.total_time_steps + 1, dtype=torch.float64)
        for _ in range(num_samples):
            sample_input = self.input_gen.generate_input_tensor(generation_steps)
            sample_target = self.target_gen.generate_target_tensor(generation_steps, sample_input)
            sample_input, sample_target = self._post_process_data_matrices(sample_input, sample_target)
            # for transformer training, we are interested to predict Y_{t+1} with input x_t, but transformer
            # datasets align input to desired output i.e.
            # y_1   y_2     y_3     y_4
            # x_0   x_1     x_2     x_3
            # Therefore, we should shift the target matrix by one time step to bigger ts and we need to trim the final
            # x to make them the same size
            sample_input = sample_input[:-1, :]
            sample_target = sample_target[1:, :]
            data.append(sample_input)
            targets.append(sample_target)
        # every input and output in our dataloader has the shape (self.total_time_steps - 1 , x_dim/y_dim)
        dataset: BaseDataset = TimeSeriesTorchDataset(data, targets)

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
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for i in range(self.input_matrix.shape[1]):
            plt.plot(self.time_axis, self.input_matrix[:, i], label=f"Input: $x_{i+1}$", linestyle="-", linewidth=2.5)

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=title_font)
        plt.ylabel("Input", fontdict=axis_font)
        plt.title("Input Features", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 14}, labelspacing=0)
        plt.tight_layout(pad=0.5)

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
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Plot target y
        for i in range(self.target_matrix.shape[1]):
            plt.plot(
                self.time_axis, self.target_matrix[:, i], label=f"Target: $y_{i+1}$", linestyle="-", linewidth=2.5
            )

        # Plot server's prediction
        for i in range(server_matrix.shape[1]):
            plt.plot(
                self.time_axis,
                server_matrix[:, i].detach().numpy(),
                label=f"Prediction: Server $\\hat{{Y}}_{i+1}$",
                linestyle=":",
                linewidth=2.5,
            )

        if game_played:
            if show_points:
                # Display synchronization steps as points
                for i in range(server_matrix.shape[1]):
                    T_indices = [i * T for i in range(1, int(self.total_time_steps / T))]
                    T_values = [server_matrix[j, i].detach().numpy().item() for j in T_indices]
                    plt.scatter(
                        T_indices,
                        T_values,
                        s=75,
                        marker="o",
                        facecolors="r",
                        edgecolors="r",
                        zorder=3,
                    )
            else:
                # Display synchronization steps as vertical lines instead
                for j in range(1, int(self.total_time_steps / T)):
                    label = "Nash Game Played" if j == 1 else None
                    plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

        if game_played:
            game_status = ""
        else:
            game_status = "No "

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Time-Series Values", fontdict=axis_font)
        plt.title(f"Server Predictions ({game_status}Nash Game)", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, labelspacing=0)
        plt.tight_layout(pad=0.5)

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
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
                    label=f"Prediction: Server $\\hat{{Y}}_{i+1}$",
                    linestyle="dashdot",
                    linewidth=2.5,
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
                plt.plot(
                    self.time_axis, self.target_matrix[:, i], label=f"Target: $y_{i+1}$", linestyle="-", linewidth=2.5
                )

        if show_input:
            for i in range(self.input_matrix.shape[1]):
                plt.plot(
                    self.time_axis, self.input_matrix[:, i], label=f"Input: $x_{i+1}$", linestyle="--", linewidth=2.5
                )

        for client in range(int(plot_info["num_clients"])):
            for dim in range(clients_pred_matrix.shape[2]):
                plt.plot(
                    self.time_axis,
                    clients_pred_matrix[:, client, dim],
                    label=f"Prediction: $\\mathregular{{Client}}_{client}$ $\\hat{{Y}}_{dim+1}$",
                    linestyle=":",
                    linewidth=2.5,
                )

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Time-Series Values", fontdict=axis_font)
        plt.title("Individual Client Predictions", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, labelspacing=0)
        plt.tight_layout(pad=0.5)

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

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
                label=f"Weight: $\\mathregular{{Client}}_{client}$",
                linestyle="-",
                linewidth=2.5,
            )

        if game_played:
            if show_lines:
                # Highlight synchronization time steps with vertical lines
                for j in range(1, int((self.total_time_steps) / T)):
                    label = "Nash Game Played" if j == 1 else None
                    plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

            if show_points:
                # Highlight synchronization steps with points
                for client in range(int(plot_info["num_clients"])):
                    T_indices = [i * T for i in range(1, int((self.total_time_steps) / T))]
                    T_values = [mixture_weights[j, client] for j in T_indices]
                    plt.scatter(
                        T_indices,
                        T_values,
                        marker="o",
                        s=75,
                        facecolors="r",
                        edgecolors="r",
                        zorder=3,
                    )

        if game_played:
            game_status = ""
        else:
            game_status = "No "

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Mixture Weight", fontdict=axis_font)
        plt.title(f"Mixture Weights ({game_status}Nash Game)", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 14}, labelspacing=0)
        plt.tight_layout(pad=0.55)
        plt.savefig(plot_path)

        plt.close()

    def visualize_squared_error_histogram(
        self,
        server_prediction: List[torch.Tensor],
        plot_path: str,
        plot_info: Dict[str, Any],
        game_played: bool = False,
    ) -> None:
        """
        Saves a histogram showing the error distribution of the predictions made by the server
        """
        server_matrix = torch.stack(server_prediction, dim=0).squeeze(-1)
        assert server_matrix.shape == (self.total_time_steps, self.target_matrix.shape[1]), {
            f"Error:server output matrix has a shape {server_matrix.shape},\
                but it should be{(self.total_time_steps, self.target_matrix.shape[1])}"
        }

        squared_error = (server_matrix - self.target_matrix) ** 2
        squared_error = squared_error.flatten()
        plt.hist(squared_error, bins=10)

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        game_status = "" if game_played else "No "

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Squared Error Bins", fontdict=axis_font)
        plt.ylabel("Squared Error Counts", fontdict=axis_font)
        plt.title(f"Histogram of Squared Errors ({game_status}Nash Game)", fontdict=title_font)

        plt.tight_layout(pad=0.5)

        plt.savefig(plot_path)

        plt.close()

    def visualize_server_squared_errors(
        self,
        server_prediction: List[torch.Tensor],
        plot_path: str,
        game_played: bool = False,
        T: int = 0,
        show_points: Optional[bool] = False,
        plot_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves a time series plot showing the server prediction squared errors
        """

        if game_played:
            assert T > 0, "Error: if the game is played, T should be greater than zero."

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        server_matrix = torch.stack(server_prediction, dim=0).squeeze(-1)
        assert server_matrix.shape == (self.total_time_steps, self.target_matrix.shape[1]), {
            f"Error:server output matrix has a shape {server_matrix.shape},\
                but it should be{(self.total_time_steps, self.target_matrix.shape[1])}"
        }

        squared_error = (server_matrix - self.target_matrix) ** 2

        # Plot server's prediction squared errors
        for i in range(server_matrix.shape[1]):
            plt.plot(
                self.time_axis,
                squared_error[:, i].detach().numpy(),
                label=f"$(\\hat{{Y}}_{i+1} - y_{i+1})^2$",
                linestyle="-",
                linewidth=2.5,
            )

        if game_played:
            if show_points:
                # Display synchronization steps as points
                for i in range(server_matrix.shape[1]):
                    T_indices = [i * T for i in range(1, int(self.total_time_steps / T))]
                    T_values = [squared_error[j, i].detach().numpy().item() for j in T_indices]
                    plt.scatter(
                        T_indices,
                        T_values,
                        s=75,
                        marker="o",
                        facecolors="r",
                        edgecolors="r",
                        zorder=3,
                    )
            else:
                # Display synchronization steps as vertical lines instead
                for j in range(1, int(self.total_time_steps / T)):
                    label = "Nash Game Played" if j == 1 else None
                    plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        game_status = "" if game_played else "No "

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Squared Error Values", fontdict=axis_font)
        plt.title(f"Server Squared Errors ({game_status}Nash Game)", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, labelspacing=0)
        plt.tight_layout(pad=0.5)

        plt.savefig(plot_path)

        plt.close()

    def visualize_client_squared_errors(
        self,
        client_predictions: List[torch.Tensor],
        plot_path: str,
        plot_info: Dict[str, Any],
        game_played: bool = False,
        T: int = 0,
        show_points: Optional[bool] = False,
    ) -> None:
        """
        Saves a time series plot showing the client prediction squared errors
        """

        if game_played:
            assert T > 0, "Error: if the game is played, T should be greater than zero."

        clients_pred_matrix = torch.stack(client_predictions, dim=0)
        assert clients_pred_matrix.shape == (
            self.total_time_steps,
            plot_info["num_clients"],
            self.target_matrix.shape[1],
        ), {
            f"Error: client prediction matrix shape is {clients_pred_matrix.shape},\
            but it should be {(self.total_time_steps, plot_info['num_clients'], self.target_matrix.shape[1])}"
        }

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for client in range(int(plot_info["num_clients"])):
            for dim in range(clients_pred_matrix.shape[2]):
                squared_error = (clients_pred_matrix[:, client, dim] - self.target_matrix[:, dim]) ** 2
                plt.plot(
                    self.time_axis,
                    squared_error,
                    label=f"$\\mathregular{{Client}}_{client}$: $(\\hat{{Y}}_{dim+1} - y_{dim+1})^2$",
                    linestyle="-",
                    linewidth=2.5,
                )

                if game_played and show_points:
                    # Display synchronization steps as points
                    T_indices = [i * T for i in range(1, int(self.total_time_steps / T))]
                    T_values = [squared_error[j].detach().numpy().item() for j in T_indices]
                    plt.scatter(
                        T_indices,
                        T_values,
                        s=75,
                        marker="o",
                        facecolors="r",
                        edgecolors="r",
                        zorder=3,
                    )

        if game_played and not show_points:
            # Display synchronization steps as vertical lines instead
            for j in range(1, int(self.total_time_steps / T)):
                label = "Nash Game Played" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

        if plot_info is not None:
            text_content = ""
            num_items = 0
            for key, value in plot_info.items():
                text_content += f"{key}: {value},"
                num_items += 1
                if num_items % 6 == 0:
                    text_content += "\n"
            plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
            plt.subplots_adjust(bottom=0.2)

        game_status = "" if game_played else "No "

        title_font = {"family": "helvetica", "weight": "bold", "size": 20}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
        plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Squared Error Values", fontdict=axis_font)
        plt.title(f"Client Squared Errors ({game_status}Nash Game)", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, labelspacing=0)
        plt.tight_layout(pad=0.5)

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
