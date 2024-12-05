import json
from enum import Enum
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import yaml

from fedmoe.datasets.brownian_motion_dataset import BrownianSequenceAddition, TimeSeriesBrownianTarget
from fedmoe.datasets.logistic_map_dataset import TimeSeriesLogisticMap
from fedmoe.datasets.periodic_dataset import TimeInputPeriodic, TimeSeriesPeriodic
from fedmoe.datasets.simple_datasets import TimeSeriesLinearLine, TimeSeriesQuadratic, TimeSeriesSineSignal
from fedmoe.datasets.time_series_data import TimeSeries2DXY, TimeSeriesData


class DataOptions(Enum):
    PERIODIC_SIGNAL: str = "periodic_signal"
    TIME_INPUT_PERIODIC = "time_input_periodic"
    LOGISTIC_MAP = "logistic_map"
    SINE_SIGNAL = "sine_signal"
    QUADRATIC_DATA = "quadratic_data"
    LINEAR_LINE = "linear_line"
    HORIZONTAL_LINE = "horizontal_line"
    SIMPLE_BROWNIAN = "simple_brownian"
    BROWNIAN_ADDITION = "brownian_addition"
    XY_2D = "2dxy"
    BOC_EXCHANGE = "boc_exchange"
    TRANSFORMER_TEMPERATURE = "transformer_temperature"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(dataset_name: str, total_rounds: int) -> TimeSeriesData:
    # Load data
    try:
        dataset_option = DataOptions(dataset_name)
        print(f"Data option is set to: {dataset_option}")
    except ValueError:
        raise ValueError(f"Invalid data name in config: {dataset_name}")

    if dataset_option == DataOptions.PERIODIC_SIGNAL:
        return TimeSeriesPeriodic(total_time_steps=total_rounds)
    elif dataset_option == DataOptions.LOGISTIC_MAP:
        return TimeSeriesLogisticMap(total_time_steps=total_rounds)
    elif dataset_option == DataOptions.HORIZONTAL_LINE:
        # In this dataset, input is the time_step, and output is always b (a is zero).
        return TimeSeriesLinearLine(total_time_steps=total_rounds, a=0.0, b=0.5)
    elif dataset_option == DataOptions.LINEAR_LINE:
        return TimeSeriesLinearLine(total_time_steps=total_rounds, a=3.0, b=2.0)
    elif dataset_option == DataOptions.QUADRATIC_DATA:
        return TimeSeriesQuadratic(total_time_steps=total_rounds, a=2.0, b=-1.0, c=1.0)
    elif dataset_option == DataOptions.SINE_SIGNAL:
        return TimeSeriesSineSignal(total_time_steps=total_rounds)
    elif dataset_option == DataOptions.XY_2D:
        return TimeSeries2DXY(total_time_steps=total_rounds)
    elif dataset_option == DataOptions.SIMPLE_BROWNIAN:
        return TimeSeriesBrownianTarget(
            total_time_steps=total_rounds, n_brownian_trajectories=5, mu=1.0, sigma=1.0, offset=0.5
        )
    elif dataset_option == DataOptions.BROWNIAN_ADDITION:
        return BrownianSequenceAddition(
            total_time_steps=total_rounds, n_brownian_trajectories=3, mu=1.0, sigma=1.0, offset=0.0
        )
    elif dataset_option == DataOptions.TIME_INPUT_PERIODIC:
        return TimeInputPeriodic(total_time_steps=total_rounds)
    else:
        raise ValueError(f"dataset name {dataset_name} is not valid. See DataOptions.")


def save_to_json(tensors_to_save: Dict[str, List[torch.Tensor]], path: str) -> None:
    # Creating a new dict to avoid mypy error.
    lists_to_save = {}
    for data_name, data_list in tensors_to_save.items():
        lists_to_save[data_name] = [torch_tensor.tolist() for torch_tensor in data_list]

    with open(f"{path}/data.json", "w") as f:
        json.dump(lists_to_save, f)


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
