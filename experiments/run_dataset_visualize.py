"""This script is used to visualize the datasets."""

import sys

from experiments.utils import load_data
from fedmoe.datasets.fedmoe_datasets.boc_rates import BankOfCanadaExchangeRates, ExchangeRates


if len(sys.argv) != 3:
    print("Usage: python run_dataset_visualize.py <dataset_name> <total_rounds>")
    sys.exit(1)

dataset_name = sys.argv[1]
total_rounds = int(sys.argv[2])

data = load_data(dataset_name, total_rounds)
assert hasattr(data, "visualize"), f"Dataset {dataset_name} does not have visualize method"
if dataset_name == "ett_data":
    data.visualize(visualize_only_main_inputs=True)
elif dataset_name == "boc_exchange":
    inputs = [
        ExchangeRates.AUD_CLOSE,
        ExchangeRates.EUR_CLOSE,
        ExchangeRates.GBP_CLOSE,
        ExchangeRates.CHF_CLOSE,
    ]
    targets = [ExchangeRates.USD_CLOSE]
    input_lag = [0, 1]
    target_lag = [0, 1]
    dataset = BankOfCanadaExchangeRates(
        inputs=inputs,
        targets=targets,
        input_lags=input_lag,
        target_lags=target_lag,
    )
    dataset.cut_first_time_steps(data_sequence_length=total_rounds)
    dataset.visualize(visualize_only_main_inputs=True)
else:
    data.visualize()
