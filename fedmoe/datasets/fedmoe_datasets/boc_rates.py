from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fedmoe.datasets.time_series_data import TimeSeriesData

PandasFrame = TypeVar("pandas_frame", pd.Series, pd.DataFrame)


class ExchangeRates(Enum):
    AUD_CLOSE = "AUD_CLOSE"
    DKK_CLOSE = "DKK_CLOSE"
    EUR_CLOSE = "EUR_CLOSE"
    HKD_CLOSE = "HKD_CLOSE"
    JPY_CLOSE = "JPY_CLOSE"
    MXN_CLOSE = "MXN_CLOSE"
    NZD_CLOSE = "NZD_CLOSE"
    NOK_CLOSE = "NOK_CLOSE"
    SEK_CLOSE = "SEK_CLOSE"
    CHF_CLOSE = "CHF_CLOSE"
    GBP_CLOSE = "GBP_CLOSE"
    USD_CLOSE = "USD_CLOSE"


class BankOfCanadaExchangeRates(TimeSeriesData):
    """
    Historical daily exchange rates between CAD and multiple currencies from 2007 to 2017.
    12 currencies (CAD to X exchange rate), 3651 daily observations
    https://www.bankofcanada.ca/rates/exchange/legacy-noon-and-closing-rates/
    """

    def __init__(
        self,
        inputs: List[ExchangeRates],
        targets: List[ExchangeRates],
        input_lag: int = 1,
        target_lag: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dtype: torch.dtype = torch.float64,
        dataset_path: str = "fedmoe/datasets/assets/bank_of_canada_exchange_rates.csv",
    ) -> None:
        """
        Constructor for a Bank of Canada Exchange Rate time series dataset.

        Args:
            inputs (List[ExchangeRates]): These are the currencies in the dataset to use to help make predictions.
                The inputs and targets should be distinct. That is, target currencies should not be included here.
                If you want to include lagged values of the targets, set target_lag > 1.
            targets (List[ExchangeRates]): These are the currencies that we are trying to make predictions for. If
                multiple currencies are provided, we're predicting multiple exchange rates simultaneously
            input_lag (int, optional): How many steps back to include in the input currency exchange rates. For
                example, if input_lag is 2 and we have USD and EUR exchange rates at inputs, then at time t, we are
                attempting to make predictions for t+1 based on USD_t, USD_{t-1}, EUR_t, and EUR_{t-1}. Defaults to 1.
            target_lag (Optional[int], optional): Similar to input lag, this is how many steps backwards in time for
                target values should be include in the input. For example, if target_lag is 2 and we have USD and EUR
                exchange rates at TARGETS, then at time t, we are attempting to make predictions for t+1 for both
                currencies and we include USD_t, USD_{t-1}, EUR_t, and EUR_{t-1} in the input sequence along with other
                input values. If none, then lagged targets are not included in the input. Defaults to None.
            start_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to
                begin.  The minimum value for this argument is 2007-05-01. If None, the minimum value is provided.
                If not the minimum value and input_lag/target_lag are greater than 1, we will still look back in time
                to gather these as far as possible. When prior time stamps are not available, we set the lagged values
                to 0. Defaults to None.
            end_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to end.
                The maximum value for this argument is 2017-04-28. If None, the maximum value is provided. Defaults to
                None.
            dtype (torch.dtype, optional): Default type for any torch tensors created. Defaults to torch.float64.
            dataset_path (str): Path to the dataset csv file. By default it is assumed to exist in the datasets/assets
                folder.
        """
        assert (
            len(inputs) > 0 or target_lag is not None
        ), "No inputs specified. Either specify input features or specify a target lag"
        assert len(targets) > 0, "No targets have been specified."

        assert input_lag > 0, "Input lag must be at least 1"
        if target_lag is not None:
            assert target_lag > 0, "Target lag must be at least 1"

        self.inputs = inputs
        self.targets = targets
        self.input_lag = input_lag
        self.target_lag = target_lag
        self._verify_inputs_and_targets_are_mutually_exclusive()

        self.min_date = datetime(2007, 5, 1)
        self.max_date = datetime(2017, 4, 28)
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        assert start_date < end_date, "Start date occurs after end date. This is invalid"
        assert (
            self.min_date <= start_date <= self.max_date
        ), "Start date must occur on or after 2007-05-01 and on or before 2017-04-28"
        assert (
            self.min_date <= end_date <= self.max_date
        ), "End date must occur on or after 2007-05-01 and on or before 2017-04-28"

        self.start_date = start_date
        self.end_date = end_date
        self.dtype = dtype

        self.total_time_steps = (end_date - start_date).days + 1
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.dataset_path = dataset_path

        # Generate data set
        self.input_matrix, self.target_matrix = self._construct_dataset()

        self.x_dim = self.input_matrix.shape[1]
        self.y_dim = self.target_matrix.shape[1]

    def _construct_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_data = pd.read_csv(self.dataset_path)
        # Format the date column
        raw_data["date"] = pd.to_datetime(raw_data["date"])
        # Set date column as index
        raw_data = raw_data.set_index("date")

        input_tensors = []
        for lag in range(1, self.input_lag + 1):
            # Filter the data to the time period we care about
            date_time_filtered = self._get_lagged_value(lag, raw_data)
            # Get the input values based on the input currencies
            input_df = date_time_filtered.loc[:, [input.value for input in self.inputs]]
            # Create tensor
            input_tensor = torch.from_numpy(input_df.values).to(self.dtype)
            # Left pad the tensor to the correct size with zeros in each row if it's too small
            n_rows, n_columns = input_tensor.shape
            pad_length = self.total_time_steps - n_rows
            input_tensors.append(torch.cat([torch.zeros((pad_length, n_columns)), input_tensor], dim=0))

        if self.target_lag is not None:
            for lag in range(1, self.target_lag + 1):
                # Filter the data to the time period we care about
                date_time_filtered = self._get_lagged_value(lag, raw_data)
                # Get the input values based on the input currencies
                target_df = date_time_filtered.loc[:, [target.value for target in self.targets]]
                # Create tensor
                target_tensor = torch.from_numpy(target_df.values).to(self.dtype)
                # Left pad the tensor to the correct size with zeros in each row if it's too small
                n_rows, n_columns = target_tensor.shape
                pad_length = self.total_time_steps - n_rows
                input_tensors.append(torch.cat([torch.zeros((pad_length, n_columns)), target_tensor], dim=0))

        final_input_tensor = torch.cat(input_tensors, dim=1)

        date_time_filtered = raw_data.loc[str(self.start_date) : str(self.end_date)]
        target_df = date_time_filtered.loc[:, [target.value for target in self.targets]]
        final_target_tensor = torch.from_numpy(target_df.values).to(self.dtype)

        return final_input_tensor, final_target_tensor

    def _get_lagged_value(self, lag: int, df: pd.DataFrame) -> pd.DataFrame:
        assert lag >= 0, "Lag cannot be negative"
        lagged_start_date = self.start_date - timedelta(days=lag)
        lagged_end_date = self.end_date - timedelta(days=lag)
        return df.loc[str(lagged_start_date) : str(lagged_end_date)]

    def _verify_inputs_and_targets_are_mutually_exclusive(self) -> None:
        inputs_set = set(self.inputs)
        targets_set = set(self.targets)
        assert len(inputs_set.intersection(targets_set)) == 0, "Inputs and targets should not overlap"

    def visualize(self) -> None:
        n_inputs = self.input_matrix.shape[1]
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        for input_path in range(n_inputs):
            ax.plot(self.time_axis, self.input_matrix[:, input_path], linestyle="solid", linewidth=1.5)
        for target_path in range(n_targets):
            ax.plot(self.time_axis, self.target_matrix[:, target_path], linestyle="dotted", linewidth=1.5)
        ax.set_title("Exchange Rate All")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.show()


