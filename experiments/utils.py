from typing import Any, Dict

import matplotlib.pyplot as plt
import torch

from fedmoe.datasets.logistic_map_dataset import get_logistic_map_sequence
from fedmoe.datasets.periodic_dataset import get_periodic_signal_sequence


def load_data(dataset_name: str, total_rounds: int) -> torch.Tensor:
    # Load data
    if dataset_name == "periodic_signal":
        data_sequence = get_periodic_signal_sequence(n_samples=1, data_length=total_rounds + 1)
    elif dataset_name == "logistic_map":
        data_sequence = get_logistic_map_sequence(n_samples=1, data_length=total_rounds + 1)
    elif dataset_name == "linear_line":
        data_sequence = torch.Tensor([0.5 for i in range(total_rounds + 1)])
    else:
        raise ValueError("dataset name is not valid.")
    # Assumption is that the data_sequence has shape time x x_dim (1 here)
    return data_sequence.reshape(-1, 1)


def plot_sequence(
    input_sequence: torch.Tensor,
    prediction_sequence: torch.Tensor,
    T: int,
    have_sync: bool,
    plot_info: Dict[str, Any],
    plot_path: str,
    show: bool = False,
    plot_name: str = "prediction_seq",
) -> None:
    plt.plot(input_sequence, label="input data", color="gray", alpha=0.5)
    plt.plot(prediction_sequence, label="prediction", color="blue", alpha=0.5, linewidth=2)

    if have_sync:
        T_indices = [i * T for i in range(1, int((prediction_sequence.size(0) - 1) / T) + 1)]
        T_values = [prediction_sequence[i] for i in T_indices]
        plt.scatter(T_indices, T_values, color="r", marker="o", label="T")

    text_content = "\n".join([f"{key}: {value}" for key, value in plot_info.items()])
    plt.text(
        0.05,
        0.95,
        text_content,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Experiment data")
    plt.ylim((-2, 2))
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(plot_path + "/" + plot_name + ".png")


def plot_results(
    results: torch.Tensor,
    average_metric_value: float,
    T: int,
    have_sync: bool,
    plot_info: Dict[str, Any],
    plot_path: str,
    show: bool = False,
    plot_name: str = "error_plot",
) -> None:
    """
    This method can be used to plot a sequence of computed metrics or errors at each time-step.
    The functionality of having such a results is not yet implemented because I am not sure if it is useful.
    """
    plt.plot(results, label="Results", color="gray", alpha=0.5)
    # Add a dot on each data point
    plt.scatter(range(results.size(0)), results, color="b", marker="o")
    # Highlight synchronization steps in red
    highlighted_indices = [i * T for i in range(int((results.size(0)) / T) + 1)]
    highlighted_values = torch.Tensor([results[i] for i in highlighted_indices])
    plt.scatter(highlighted_indices, highlighted_values, color="r", marker="o", label="Highlighted Points")

    if have_sync:
        T_indices = [i * T for i in range(1, int((results.size(0) - 1) / T) + 1)]
        T_values = [results[i] for i in T_indices]
        plt.scatter(T_indices, T_values, color="r", marker="o", label="T")

    text_content = "\n".join([f"{key}: {value}" for key, value in plot_info.items()])
    plt.text(
        0.05,
        0.95,
        text_content,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.axhline(y=average_metric_value, color="r", linestyle="--", label="Average Metric line")

    plt.xlabel("Time")
    plt.ylabel("Metric (Error) Value")
    plt.title("Experiment Results")
    plt.xticks(highlighted_indices)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.ylim((0, 2))
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(plot_path + "/" + plot_name + ".png")
