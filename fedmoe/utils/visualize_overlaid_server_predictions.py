import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import MaxNLocator


sns.set_style("whitegrid")


def visualize_relative_server_squared_errors(
    time_axis: torch.Tensor,
    target_matrix: torch.Tensor,
    game_server_predictions: torch.Tensor,
    non_game_server_predictions: torch.Tensor,
    plot_path: str,
    render_line: bool = False,
) -> None:
    plt.rcParams["figure.figsize"] = [10, 3]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot target y
    for i in range(target_matrix.shape[1]):
        sns.lineplot(
            x=time_axis,
            y=target_matrix[:, i],
            label=f"Target: $y_{i + 1}$",
            linestyle="-",
            linewidth=2.5,
        )

    # Plot server's prediction
    for i in range(game_server_predictions.shape[1]):
        sns.lineplot(
            x=time_axis,
            y=game_server_predictions[:, i].detach().numpy(),
            label=f"Nash Server $\\hat{{Y}}_{i + 1}$",
            linestyle=":",
            linewidth=2.5,
        )

    for i in range(non_game_server_predictions.shape[1]):
        sns.lineplot(
            x=time_axis,
            y=non_game_server_predictions[:, i].detach().numpy(),
            label=f"Non-Nash Server $\\hat{{Y}}_{i + 1}$",
            linestyle=":",
            linewidth=2.5,
        )

    title_font = {"family": "helvetica", "weight": "bold", "size": 20}
    axis_font = {"family": "helvetica", "weight": "bold", "size": 18}
    plt.xticks(fontname="helvetica", fontsize=14, fontweight="bold")
    plt.yticks(fontname="helvetica", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step", fontdict=axis_font)
    plt.ylabel("Time-Series Values", fontdict=axis_font)
    plt.title("Server Predictions vs. Target", fontdict=title_font)

    plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 12}, labelspacing=0)
    plt.tight_layout(pad=1)

    plt.savefig(plot_path, format="pdf")

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
        os.path.join(output_dir, "overlaid_server_predictions.pdf"),
        render_line=True,
    )
