import random
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.time_series_data import TimeSeriesData, TimeSeriesTorchDataset

sns.set_style("whitegrid")
torch.set_default_dtype(torch.float64)


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
        input_lags: List[int],
        target_lags: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dtype: torch.dtype = torch.float64,
        dataset_path: str = "fedmoe/datasets/assets/bank_of_canada_exchange_rates.csv",
    ) -> None:
        """
        Constructor for a Bank of Canada Exchange Rate time series dataset.
        NOTE: By convention, at time step t, we are making predictions for y_{t+1} using x_t. So x_t can encode
        currency exchange rates before or AT time t. As such, lags of 0 for input creation are perfectly valid
        for both input_lags and target_lags.

        For example: If targets is [USD] and inputs is [AUD] with lags of [0, 1] for both input and target then the
        tensors look like
        Y = [USD_t,         USD_{t+1},  USD_{t+2},  ..., USD_n]^T

        X = [[USD_t,        USD_{t+1},  USD_{t+2},  ..., USD_n],
             [USD_{t-1},    USD_t,      USD_{t+1},  ..., USD_{n-1}],
             [AUD_t,        AUD_{t+1},  AUD_{t+2},  ...,  AUD_n],
             [AUD_{t-1},    AUD_t,      AUD_{t+1},  ..., AUD_{n-1}]]^T

        In this setup, X[0, :] = [USD_t, USD_{t-1}, AUD_t, AUD_{t-1}] would be used to predict Y[1] = USD_{t+1}

        Args:
            inputs (List[ExchangeRates]): These are the currencies in the dataset to use to help make predictions.
                The inputs and targets should be distinct. That is, target currencies should not be included here.
                If you want to include lagged values of the targets, set target_lag.
            targets (List[ExchangeRates]): These are the currencies that we are trying to make predictions for. If
                multiple currencies are provided, we're predicting multiple exchange rates simultaneously
            input_lags (List[int]): List of steps backward in input that should be included in the input features.
                For example, if input_lag is [1, 2] and we have USD and EUR exchange rates at inputs, then at time t,
                then x_t contains USD_{t-1}, USD_{t-2}, EUR_{t-1}, and EUR_{t-2}.
            target_lags (Optional[List[int]], optional): Similar to input lag, this is a list of steps backward in
                target values should be include in the input. For example, if target_lag is [1, 2] and we have USD and
                EUR exchange rates at TARGETS, then at time t, x_t for y_t contains USD_{t-1}, USD_{t-2}, EUR_{t-1},
                EUR_{t-2} in the input sequence  along with other input values.  If none, then lagged targets are not
                included in the input.
            start_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to
                begin.  The minimum value for this argument is 2007-05-01. If None, the minimum value is used.
                If not the minimum value and input_lag/target_lag are greater than 1, we will still look back in time
                to gather these as far as possible. When prior time stamps are not available, we set the lagged values
                to 0. Defaults to None.
            end_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to end.
                The maximum value for this argument is 2017-04-28. If None, the maximum value is used. Defaults to
                None.
            dtype (torch.dtype, optional): Default type for any torch tensors created. Defaults to torch.float64.
            dataset_path (str): Path to the dataset csv file. By default it is assumed to exist in the datasets/assets
                folder.
        """
        assert (
            len(inputs) > 0 or target_lags is not None
        ), "No inputs specified. Either specify input features or specify a target lag"
        assert len(targets) > 0, "No targets have been specified."

        for input_lag in input_lags:
            assert input_lag >= 0, "Input lag must be at least 0"
        if target_lags is not None:
            for target_lag in target_lags:
                assert target_lag >= 0, "Target lag must be at least 0"

        self.inputs = inputs
        self.targets = targets
        self.input_lags = input_lags
        self.target_lags = target_lags
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
        for lag in self.input_lags:
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

        if self.target_lags is not None:
            for lag in self.target_lags:
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

        # Type is ignored here since .loc does indeed take these strings in the case of a datetime index
        date_time_filtered = raw_data.loc[str(self.start_date) : str(self.end_date)]  # type: ignore
        target_df = date_time_filtered.loc[:, [target.value for target in self.targets]]
        final_target_tensor = torch.from_numpy(target_df.values).to(self.dtype)

        return final_input_tensor, final_target_tensor

    def _get_lagged_value(self, lag: int, df: pd.DataFrame) -> pd.DataFrame:
        assert lag >= 0, "Lag cannot be negative"
        lagged_start_date = self.start_date - timedelta(days=lag)
        lagged_end_date = self.end_date - timedelta(days=lag)
        # Type is ignored here since .loc does indeed take these strings in the case of a datetime index
        return df.loc[str(lagged_start_date) : str(lagged_end_date)]  # type: ignore

    def _verify_inputs_and_targets_are_mutually_exclusive(self) -> None:
        inputs_set = set(self.inputs)
        targets_set = set(self.targets)
        assert len(inputs_set.intersection(targets_set)) == 0, "Inputs and targets should not overlap"

    def visualize(self, visualize_only_main_inputs: bool = False) -> None:
        """
        Args:
            visualize_only_main_inputs (bool, optional): Is set to true only visualizes the six main inputs
            with their labels.
        """
        if visualize_only_main_inputs:
            n_inputs = len(self.inputs)
        else:
            n_inputs = self.input_matrix.shape[1]
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(20, 8))
        for target_path in range(n_targets):
            sns.lineplot(
                x=self.time_axis,
                y=self.target_matrix[:, target_path],
                label="USD",
                ax=ax,
                linestyle="solid",
                linewidth=3,
            )

        for input_path in range(n_inputs):
            sns.lineplot(
                x=self.time_axis,
                y=self.input_matrix[:, input_path],
                label=self.inputs[input_path].value.replace("_CLOSE", "") if visualize_only_main_inputs else None,
                ax=ax,
                linestyle="solid",
                linewidth=3,
            )

        title_font = {"family": "helvetica", "weight": "bold", "size": 35}
        axis_font = {"family": "helvetica", "weight": "bold", "size": 35}
        plt.xticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.yticks(fontname="helvetica", fontsize=30, fontweight="bold")
        plt.xlabel("Time Step", fontdict=axis_font)
        plt.ylabel("Exchange Rate Relative to CAD", fontdict=axis_font)
        plt.title("Exchange Rates for a Selection of Currencies", fontdict=title_font)

        plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 30}, loc="upper left", labelspacing=0)
        plt.tight_layout(pad=0.5)

        plt.savefig("boc_exchange_rates.pdf", format="pdf")
        plt.show()

    def cut_first_time_steps(self, data_sequence_length: int) -> None:
        """
        Shortens the time series data to the specified length.
        Keeps from the start of the time series.

        Args:
            data_sequence_length (int): the desired length of the sequence (total_time_steps).
        """
        self.original_total_time_steps = self.total_time_steps
        self.total_time_steps = data_sequence_length
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.original_input_matrix = self.input_matrix
        self.original_target_matrix = self.target_matrix
        self.input_matrix = self.input_matrix[:data_sequence_length, :]
        self.target_matrix = self.target_matrix[:data_sequence_length, :]

    def cut_at_start_index(self, data_sequence_length: int, start_index: int) -> None:
        self.original_total_time_steps = self.total_time_steps
        self.total_time_steps = data_sequence_length
        self.time_axis = torch.arange(0, self.total_time_steps)
        self.original_input_matrix = self.input_matrix
        self.original_target_matrix = self.target_matrix
        self.input_matrix = self.input_matrix[start_index : start_index + data_sequence_length, :]
        self.target_matrix = self.target_matrix[start_index : start_index + data_sequence_length, :]

    def maybe_random_cut_time_steps(
        self, data_sequence_length: int, start_index: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly cuts the original time series data to the specified length.
        If start_index is given start from that index and get the sequence.
        The main usage of this function is for generating samples for dataloader.
        See self.get_dataloader
        """
        if start_index is None:
            # Generate a random number between 0 and self.original_total_time_steps - data_sequence_length - 1.
            start_index = random.randint(0, self.original_total_time_steps - data_sequence_length - 1)
        selected_input_matrix = self.original_input_matrix[start_index : start_index + data_sequence_length, :]
        selected_target_matrix = self.original_target_matrix[start_index : start_index + data_sequence_length, :]
        return selected_input_matrix, selected_target_matrix

    def get_dataloader(self, num_samples: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        This function can be used to generate data loaders for train or validation, which are mainly
        used to pre-train the transformer model.
        """
        # Generate new data samples
        data: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for _ in range(num_samples):
            # Since the sampled sequence will be trimmed by one step later, we generate a longer sequence.
            sample_input, sample_target = self.maybe_random_cut_time_steps(self.total_time_steps + 1)
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
        # every input and output in our dataloader has the shape (self.total_time_steps, x_dim/y_dim)
        dataset: BaseDataset = TimeSeriesTorchDataset(data, targets)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        # Each item (input or output) in the data_loader will have a shape of (batch_size, time_steps, dim)
        return data_loader
