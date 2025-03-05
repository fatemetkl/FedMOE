import argparse
import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_style("whitegrid")

def visualize_input(
    time_axis: torch.Tensor,
    input_matrix: torch.Tensor,
    plot_path: str,
    plot_info: Optional[Dict[str, Any]] = None,
    show_plot_info: bool = False,
) -> None:
    """
    Saves a plot showing the input_matrix.
        Args:
            plot_path (str): the plot path (including name and location) to save the plot
            plot_info: (Optional[Dict[str, Any]]): additional information of the experiment setting to be
            added to the plot.
    """
    plt.rcParams["figure.figsize"] = [10, 5]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(input_matrix.shape[1]):
        sns.lineplot(x=time_axis, y=input_matrix[:, i], label=f"Input: $x_{i+1}$", linestyle="-", linewidth=2.5)

    if plot_info is not None and show_plot_info:
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
    time_axis: torch.Tensor,
    target_matrix: torch.Tensor,
    server_matrix: torch.Tensor,
    plot_path: str,
    game_played: bool = False,
    T: int = 0,
    show_points: Optional[bool] = False,
    show_lines: Optional[bool] = False,
    plot_info: Optional[Dict[str, Any]] = None,
    show_plot_info: bool = False,
) -> None:
    if game_played:
        assert T > 0, "Error: if the game is played, T should be greater than zero."

    total_time_steps = target_matrix.shape[0]

    assert server_matrix.shape == (total_time_steps, target_matrix.shape[1]), {
        f"Error:server output matrix has a shape {server_matrix.shape},\
            but it should be{(total_time_steps, target_matrix.shape[1])}"
    }

    plt.rcParams["figure.figsize"] = [10, 4]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot target y
    for i in range(target_matrix.shape[1]):
        sns.lineplot(x=time_axis, y=target_matrix[:, i], label=f"Target: $y_{i+1}$", linestyle="-", linewidth=2.5)

    # Plot server's prediction
    for i in range(server_matrix.shape[1]):
        sns.lineplot(
            x=time_axis,
            y=server_matrix[:, i].detach().numpy(),
            label=f"Prediction: Server $\\hat{{Y}}_{i+1}$",
            linestyle=":",
            linewidth=3.5,
        )

    if game_played:
        if show_points:
            # Display synchronization steps as points
            for i in range(server_matrix.shape[1]):
                T_indices = [i * T for i in range(1, int(total_time_steps / T))]
                T_values = [server_matrix[j, i].detach().numpy().item() for j in T_indices]
                sns.scatterplot(
                    x=T_indices,
                    y=T_values,
                    s=75,
                    marker="o",
                    facecolors="r",
                    edgecolors="r",
                    zorder=3,
                )
        elif show_lines:
            # Display synchronization steps as vertical lines instead
            for j in range(1, int(total_time_steps / T)):
                label = "Nash Game Played" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

    if game_played:
        game_status = ""
    else:
        game_status = "No "

    if plot_info is not None and show_plot_info:
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

    plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, loc="upper left", labelspacing=0)
    plt.tight_layout(pad=0.5)

    plt.savefig(plot_path, format='pdf')

    plt.close()


def visualize_clients_predictions(
    time_axis: torch.Tensor,
    input_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    clients_pred_matrix: torch.Tensor,
    plot_path: str,
    plot_info: Dict[str, Any],
    show_target: bool = True,
    show_input: bool = False,
    show_plot_info: bool = False,
) -> None:
    assert plot_info["num_clients"] is not None

    total_time_steps = target_matrix.shape[0]

    # Shape of client prediction tensor should be time x num_clients x y_dim
    assert clients_pred_matrix.shape == (
        total_time_steps,
        plot_info["num_clients"],
        target_matrix.shape[1],
    ), {
        f"Error: client prediction matrix shape is {clients_pred_matrix.shape},\
        but it should be {(total_time_steps, plot_info['num_clients'], target_matrix.shape[1])}"
    }

    plt.rcParams["figure.figsize"] = [10, 4]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if show_target:
        for i in range(target_matrix.shape[1]):
            sns.lineplot(x=time_axis, y=target_matrix[:, i], label=f"Target: $y_{i+1}$", linestyle="-", linewidth=2.5)

    if show_input:
        for i in range(input_matrix.shape[1]):
            sns.lineplot(x=time_axis, y=input_matrix[:, i], label=f"Input: $x_{i+1}$", linestyle="--", linewidth=2.5)

    for client in range(int(plot_info["num_clients"])):
        for dim in range(clients_pred_matrix.shape[2]):
            sns.lineplot(
                x=time_axis,
                y=clients_pred_matrix[:, client, dim],
                label=f"Prediction: $\\mathregular{{Client}}_{client}$ $\\hat{{Y}}_{dim+1}$",
                linestyle=":",
                linewidth=3.5,
            )

    if plot_info is not None and show_plot_info:
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

    plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, loc="upper left", labelspacing=0)
    plt.tight_layout(pad=0.5)

    plt.savefig(plot_path, format='pdf')

    plt.close()


def visualize_mixture_weights(
    time_axis: torch.Tensor,
    mixture_weights: torch.Tensor,
    plot_path: str,
    plot_info: Dict[str, Any],
    game_played: bool = False,
    T: int = 0,
    show_points: bool = False,
    show_lines: bool = False,
    show_plot_info: bool = False,
) -> None:
    assert plot_info["num_clients"] is not None
    if game_played:
        assert T > 0, "Error: if the game is played, T should be greater than zero."

    total_time_steps = time_axis.shape[0]

    assert mixture_weights.shape == (
        total_time_steps - 1,
        plot_info["num_clients"],
        1,
    ), f"Error: mixture_weights.shape is {mixture_weights.shape}, but should be (time_steps - 1 , num_clients, 1)"

    plt.rcParams["figure.figsize"] = [10, 6]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for client in range(int(plot_info["num_clients"])):
        sns.lineplot(
            x=time_axis[:-1],
            y=mixture_weights[:, client, 0],
            # label=f"Weight: $\\mathregular{{Client}}_{client}$",
            linestyle="-",
            linewidth=3,
        )

    if game_played:
        if show_lines:
            # Highlight synchronization time steps with vertical lines
            for j in range(1, int((total_time_steps) / T)):
                label = "Nash Game Played" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

        if show_points:
            # Highlight synchronization steps with points
            for client in range(int(plot_info["num_clients"])):
                T_indices = [i * T for i in range(1, int((total_time_steps) / T))]
                T_values = [mixture_weights[j, client] for j in T_indices]
                sns.scatterplot(
                    x=T_indices,
                    y=T_values,
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

    if plot_info is not None and show_plot_info:
        text_content = ""
        num_items = 0
        for key, value in plot_info.items():
            text_content += f"{key}: {value},"
            num_items += 1
            if num_items % 6 == 0:
                text_content += "\n"
        plt.text(0.5, -0.2, text_content.rstrip(",\n"), ha="center", va="top", transform=plt.gca().transAxes)
        plt.subplots_adjust(bottom=0.2)

    title_font = {"family": "helvetica", "weight": "bold", "size": 38}
    axis_font = {"family": "helvetica", "weight": "bold", "size": 34}
    plt.xticks(fontname="helvetica", fontsize=26, fontweight="bold")
    plt.yticks(fontname="helvetica", fontsize=26, fontweight="bold")
    plt.xlabel("Time Step", fontdict=axis_font)
    plt.ylabel("Mixture Weight", fontdict=axis_font)
    plt.title(f"Mixture Weights ({game_status}Nash Game)", fontdict=title_font)

    # plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 28}, labelspacing=0)
    plt.tight_layout(pad=0.55)
    plt.savefig(plot_path, format='pdf')

    plt.close()


def visualize_squared_error_histogram(
    target_matrix: torch.Tensor,
    server_matrix: torch.Tensor,
    plot_path: str,
    plot_info: Dict[str, Any],
    game_played: bool = False,
    show_plot_info: bool = False,
) -> None:
    """
    Saves a histogram showing the error distribution of the predictions made by the server
    """
    total_time_steps = target_matrix.shape[0]
    assert server_matrix.shape == (total_time_steps, target_matrix.shape[1]), {
        f"Error:server output matrix has a shape {server_matrix.shape},\
            but it should be{(total_time_steps, target_matrix.shape[1])}"
    }

    plt.rcParams["figure.figsize"] = [6.4, 4.8]
    squared_error = (server_matrix - target_matrix) ** 2
    squared_error = squared_error.flatten()
    plt.hist(squared_error, bins=10)

    if plot_info is not None and show_plot_info:
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

    plt.savefig(plot_path, format='pdf')

    plt.close()


def visualize_server_squared_errors(
    time_axis: torch.Tensor,
    target_matrix: torch.Tensor,
    server_matrix: torch.Tensor,
    plot_path: str,
    game_played: bool = False,
    T: int = 0,
    show_points: Optional[bool] = False,
    show_lines: Optional[bool] = False,
    plot_info: Optional[Dict[str, Any]] = None,
    show_plot_info: bool = False,
) -> None:
    """
    Saves a time series plot showing the server prediction squared errors
    """

    if game_played:
        assert T > 0, "Error: if the game is played, T should be greater than zero."

    total_time_steps = target_matrix.shape[0]

    assert server_matrix.shape == (total_time_steps, target_matrix.shape[1]), {
        f"Error:server output matrix has a shape {server_matrix.shape},\
            but it should be{(total_time_steps, target_matrix.shape[1])}"
    }

    squared_error = (server_matrix - target_matrix) ** 2

    plt.rcParams["figure.figsize"] = [10, 5]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot server's prediction squared errors
    for i in range(server_matrix.shape[1]):
        sns.lineplot(
            x=time_axis,
            y=squared_error[:, i].detach().numpy(),
            label=f"$(\\hat{{Y}}_{i+1} - y_{i+1})^2$",
            linestyle="-",
            linewidth=2.5,
        )

    if game_played:
        if show_points:
            # Display synchronization steps as points
            for i in range(server_matrix.shape[1]):
                T_indices = [i * T for i in range(1, int(total_time_steps / T))]
                T_values = [squared_error[j, i].detach().numpy().item() for j in T_indices]
                sns.scatterplot(
                    x=T_indices,
                    y=T_values,
                    s=75,
                    marker="o",
                    facecolors="r",
                    edgecolors="r",
                    zorder=3,
                )
        elif show_lines:
            # Display synchronization steps as vertical lines instead
            for j in range(1, int(total_time_steps / T)):
                label = "Nash Game Played" if j == 1 else None
                plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

    if plot_info is not None and show_plot_info:
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

    plt.savefig(plot_path, format='pdf')

    plt.close()


def visualize_client_squared_errors(
    time_axis: torch.Tensor,
    target_matrix: torch.Tensor,
    clients_pred_matrix: torch.Tensor,
    plot_path: str,
    plot_info: Dict[str, Any],
    game_played: bool = False,
    T: int = 0,
    show_points: Optional[bool] = False,
    show_lines: Optional[bool] = False,
    show_plot_info: bool = False,
) -> None:
    """
    Saves a time series plot showing the client prediction squared errors
    """

    if game_played:
        assert T > 0, "Error: if the game is played, T should be greater than zero."

    total_time_steps = target_matrix.shape[0]

    assert clients_pred_matrix.shape == (
        total_time_steps,
        plot_info["num_clients"],
        target_matrix.shape[1],
    ), {
        f"Error: client prediction matrix shape is {clients_pred_matrix.shape},\
        but it should be {(total_time_steps, plot_info['num_clients'], target_matrix.shape[1])}"
    }

    plt.rcParams["figure.figsize"] = [10, 5]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for client in range(int(plot_info["num_clients"])):
        for dim in range(clients_pred_matrix.shape[2]):
            squared_error = (clients_pred_matrix[:, client, dim] - target_matrix[:, dim]) ** 2
            sns.lineplot(
                x=time_axis,
                y=squared_error,
                label=f"$\\mathregular{{Client}}_{client}$: $(\\hat{{Y}}_{dim+1} - y_{dim+1})^2$",
                linestyle="-",
                linewidth=2.5,
            )

            if game_played and show_points:
                # Display synchronization steps as points
                T_indices = [i * T for i in range(1, int(total_time_steps / T))]
                T_values = [squared_error[j].detach().numpy().item() for j in T_indices]
                sns.scatterplot(
                    x=T_indices,
                    y=T_values,
                    s=75,
                    marker="o",
                    facecolors="r",
                    edgecolors="r",
                    zorder=3,
                )

    if game_played and not show_points and show_lines:
        # Display synchronization steps as vertical lines instead
        for j in range(1, int(total_time_steps / T)):
            label = "Nash Game Played" if j == 1 else None
            plt.axvline(x=j * T, color="red", linestyle="--", linewidth=1.5, label=label)

    if plot_info is not None and show_plot_info:
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

    plt.savefig(plot_path, format='pdf')

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--state_json_path",
        action="store",
        type=str,
        required=True,
        help="Path to json file storing experiment state",
    )

    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="Path to directory for the visualizations",
    )
    parser.add_argument(
        "--game_played",
        action="store_true",
        help="Indicates if the game was played",
    )
    args = parser.parse_args()
    json_state_path = args.state_json_path
    output_dir = args.output_dir

    with open(json_state_path, "r") as f:
        json_state = json.load(f)

    server_predictions = torch.Tensor(json_state["server_prediction"]).squeeze(-1)
    client_predictions = torch.Tensor(json_state["clients_predictions"])
    input = torch.Tensor(json_state["input"])
    target = torch.Tensor(json_state["target"])
    mixture_weights = torch.Tensor(json_state["mixture_weights"])

    total_time_steps = target.shape[0]
    time_axis = torch.arange(0, total_time_steps, dtype=torch.float64)

    visualize_input(time_axis, input, os.path.join(output_dir, "input_plot.pdf"))

    # NOTE: These need to be manually entered along with some of the arguments below for visualizing the Nash
    # game synchronization information. The current components are just an example
    plot_info = {
        "num_clients": json_state["num_clients"],
        "Client T": json_state["client T"],
        "Game T": json_state["game T"],
        "d_z": json_state["d_z"],
        "alpha": json_state["alpha"],
        "gamma": json_state["gamma"],
        "sigma": json_state["sigma"], 
    }

    visualize_server_prediction(
        time_axis,
        target,
        server_predictions,
        os.path.join(output_dir, "server_predictions_plot.pdf"),
        game_played=args.game_played,
        T=10,
        plot_info=plot_info,
        show_points=False,
    )
    visualize_clients_predictions(
        time_axis,
        input,
        target,
        client_predictions,
        os.path.join(output_dir, "client_predictions_plot.pdf"),
        plot_info=plot_info,
        show_target=True,
    )
    visualize_mixture_weights(
        time_axis,
        mixture_weights,
        os.path.join(output_dir, "mixture_weights_plot.pdf"),
        game_played=args.game_played,
        T=10,
        plot_info=plot_info,
        show_points=False,
    )
    visualize_squared_error_histogram(
        target,
        server_predictions,
        os.path.join(output_dir, "squared_error_histogram.pdf"),
        plot_info=plot_info,
        game_played=args.game_played,
    )
    visualize_server_squared_errors(
        time_axis,
        target,
        server_predictions,
        os.path.join(output_dir, "squared_error_server.pdf"),
        game_played=args.game_played,
        T=10,
        plot_info=plot_info,
        show_points=False,
    )
    visualize_client_squared_errors(
        time_axis,
        target,
        client_predictions,
        os.path.join(output_dir, "squared_error_clients.pdf"),
        game_played=args.game_played,
        T=10,
        plot_info=plot_info,
        show_points=False,
    )
