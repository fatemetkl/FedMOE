import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator


def visualize_relative_server_squared_errors(
    time_axis: torch.Tensor,
    target_matrix: torch.Tensor,
    game_server_predictions: torch.Tensor,
    non_game_server_predictions: torch.Tensor,
    plot_path: str,
    render_line: bool = False,
) -> None:

    total_time_steps = target_matrix.shape[0]

    assert game_server_predictions.shape == (total_time_steps, target_matrix.shape[1]), {
        f"Error:server output matrix has a shape {game_server_predictions.shape},\
            but it should be{(total_time_steps, target_matrix.shape[1])}"
    }

    game_squared_error = torch.round((game_server_predictions - target_matrix) ** 2, decimals=8)
    non_game_squared_error = torch.round((non_game_server_predictions - target_matrix) ** 2, decimals=8)

    plt.rcParams["figure.figsize"] = [10, 3]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot server's prediction squared errors relative to one another
    for i in range(game_server_predictions.shape[1]):
        relative_error = non_game_squared_error[:, i] / game_squared_error[:, i]
        plt.scatter(
            time_axis,
            relative_error.detach().numpy(),
            label=f"$\\frac{{(\\hat{{Y}}^n_{i+1} - y_{i+1})^2}}{{(\\hat{{Y}}^g_{i+1} - y_{i+1})^2}}$",
        )

    ax.set_yscale("log")

    plt.plot(
        time_axis,
        torch.ones_like(time_axis),
        linestyle="--",
        linewidth=2.5,
        color="red",
    )

    title_font = {"family": "helvetica", "weight": "bold", "size": 20}
    axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
    plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
    plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step", fontdict=axis_font)
    plt.ylabel("Ratio of Squared Errors", fontdict=axis_font)
    plt.title("Ratio Squared Errors Without and With Nash Game", fontdict=title_font)

    plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 22}, loc="lower right", labelspacing=0)
    plt.tight_layout(pad=0.5)

    plt.savefig(plot_path)

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")

    parser.add_argument(
        "--game_state_json_path",
        action="store",
        type=str,
        required=True,
        help="Path to json file storing experiment state from a game run",
    )

    parser.add_argument(
        "--non_game_state_json_path",
        action="store",
        type=str,
        required=True,
        help="Path to json file storing experiment state from a non-game run",
    )

    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="Path to directory for the visualizations",
    )
    args = parser.parse_args()
    game_state_json_path = args.game_state_json_path
    non_game_state_json_path = args.non_game_state_json_path
    output_dir = args.output_dir

    with open(game_state_json_path, "r") as f:
        game_json_state = json.load(f)

    with open(non_game_state_json_path, "r") as f:
        non_game_json_state = json.load(f)

    game_server_predictions = torch.Tensor(game_json_state["server_prediction"]).squeeze(-1)
    game_target = torch.Tensor(game_json_state["target"])

    non_game_server_predictions = torch.Tensor(non_game_json_state["server_prediction"]).squeeze(-1)
    non_game_target = torch.Tensor(non_game_json_state["target"])

    assert torch.allclose(game_target, non_game_target)

    total_time_steps = game_target.shape[0]
    time_axis = torch.arange(0, total_time_steps, dtype=torch.float64)

    visualize_relative_server_squared_errors(
        time_axis,
        game_target,
        game_server_predictions,
        non_game_server_predictions,
        os.path.join(output_dir, "relative_squared_server_errors.png"),
        render_line=True,
    )
