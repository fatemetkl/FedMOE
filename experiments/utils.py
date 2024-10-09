from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import yaml
from fedmoe.datasets.time_series_data import TimeSeriesData, TimeSeries2DXY

# from experiments.experimental_data import create_linear_line, quadratic_data, sine_signal
from fedmoe.datasets.simple_datasets import TimeSeriesLinearLine, TimeSeriesQuadratic, TimeSeriesSineSignal
from fedmoe.datasets.logistic_map_dataset import TimeSeriesLogisticMap
from fedmoe.datasets.periodic_dataset import TimeSeriesPeriodic
from fedmoe.datasets.brownian_motion_dataset import TimeSeriesBrownianTarget, BrownianSequenceAddition


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(dataset_name: str, total_rounds: int) -> TimeSeriesData:
    # Load data
    if dataset_name == "periodic_signal":
        return TimeSeriesPeriodic(total_time_steps=total_rounds)
    elif dataset_name == "logistic_map":
        return TimeSeriesLogisticMap(total_time_steps=total_rounds)
    elif dataset_name == "horizontal_line":
        # In this dataset, input is the time_step, and output is always b (a is zero).
        return TimeSeriesLinearLine(total_time_steps=total_rounds, a=0.0, b=0.5)
    elif dataset_name == "linear_line":
        return TimeSeriesLinearLine(total_time_steps=total_rounds, a=3.0, b=2.0)
    elif dataset_name == "quadratic_data":
        return TimeSeriesQuadratic(total_time_steps=total_rounds, a=2.0, b=-1.0, c=1.0)
    elif dataset_name == "sine_signal":
        return TimeSeriesSineSignal(total_time_steps=total_rounds)
    elif dataset_name == "2dxy":
        return TimeSeries2DXY(total_time_steps=total_rounds)
    elif dataset_name == "simple_brownian":
        return TimeSeriesBrownianTarget(
            total_time_steps=total_rounds, n_brownian_trajectories=5, mu=1.0, sigma=1.0, offset=0.5
        )
    elif dataset_name == "brownian_addition":
        return BrownianSequenceAddition(
            total_time_steps=total_rounds, n_brownian_trajectories=3, mu=1.0, sigma=1.0, offset=0.0
        )
    else:
        raise ValueError("dataset name is not valid.")


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
    plt.ylim((torch.min(input_sequence) - 1, torch.max(input_sequence) + 1))
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
