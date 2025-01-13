import json
from enum import Enum
from typing import Any, Dict, List

import torch
import yaml

from fedmoe.datasets.brownian_motion_dataset import BrownianSequenceAddition, TimeSeriesBrownianTarget
from fedmoe.datasets.fedmoe_datasets.boc_rates import BankOfCanadaExchangeRates, ExchangeRates
from fedmoe.datasets.fedmoe_datasets.covariate_shift import CovariateShiftDataset
from fedmoe.datasets.fedmoe_datasets.transformer_temperature import InputFeatures, TransformerTemperature
from fedmoe.datasets.logistic_map_dataset import TimeSeriesLogisticMap
from fedmoe.datasets.periodic_dataset import TimeInputPeriodic, TimeSeriesPeriodic
from fedmoe.datasets.simple_datasets import TimeSeriesLinearLine, TimeSeriesQuadratic, TimeSeriesSineSignal
from fedmoe.datasets.time_series_data import TimeSeries2DXY, TimeSeriesData

torch.set_default_dtype(torch.float64)


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
    ETT = "ett_data"
    COVARIATE_SHIFT = "covariate_shift"


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
    elif dataset_option == DataOptions.BOC_EXCHANGE:
        inputs = [ExchangeRates.AUD_CLOSE]
        targets = [ExchangeRates.USD_CLOSE]
        input_lag = [0]
        target_lag = None
        dataset = BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lags=input_lag,
            target_lags=target_lag,
        )
        dataset.cut_first_time_steps(data_sequence_length=total_rounds)
        return dataset
    elif dataset_option == DataOptions.ETT:
        ett_inputs = [InputFeatures.MUFL]
        input_lag = [0]
        target_lag = None
        ett_dataset = TransformerTemperature(
            inputs=ett_inputs,
            input_lags=input_lag,
            target_lags=target_lag,
        )
        ett_dataset.cut_first_time_steps(data_sequence_length=total_rounds)
        return ett_dataset
    elif dataset_option == DataOptions.COVARIATE_SHIFT:
        return CovariateShiftDataset(total_time_steps=total_rounds)
    else:
        raise ValueError(f"dataset name {dataset_name} is not valid. See DataOptions.")


def save_output_json(
    tensors_to_save: Dict[str, List[torch.Tensor]], path: str, dict_to_save: Dict[str, Any] | None
) -> None:
    # Creating a new dict to avoid mypy error.
    lists_to_save = {}
    for data_name, data_list in tensors_to_save.items():
        lists_to_save[data_name] = [torch_tensor.tolist() for torch_tensor in data_list]
    if dict_to_save is not None:
        lists_to_save.update(dict_to_save)
    with open(f"{path}/data.json", "w") as f:
        json.dump(lists_to_save, f)
