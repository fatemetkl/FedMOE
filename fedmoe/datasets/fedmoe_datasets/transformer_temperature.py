import random
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.datasets.time_series_data import TimeSeriesData, TimeSeriesTorchDataset

torch.set_default_dtype(torch.float64)


class InputFeatures(Enum):
    HUFL = "HUFL"
    HULL = "HULL"
    MUFL = "MUFL"
    MULL = "MULL"
    LUFL = "LUFL"
    LULL = "LULL"


class TransformerTemperature(TimeSeriesData):
    """
    Hourly oil temperatures and various readings for a electricity transformer.
    Readings are taking from 2016-07-01 00:00:00 to 2018-06-26 19:00:00.
    This dataset was first introduced in the Informer paper
    https://github.com/zhouhaoyi/ETDataset
    The target is always the 1D value of oil temperature (OT column in the csv)
    """

    def __init__(
        self,
        inputs: List[InputFeatures],
        input_lags: List[int],
        target_lags: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dtype: torch.dtype = torch.float64,
        dataset_path: str = "fedmoe/datasets/assets/ETTh1.csv",
    ) -> None:
        """
        Constructor for an Electric Transformer temperature time series dataset. Note that this dataset only has one
        target, the oil temperature of the transformer. It corresponds to column OT in the csv file and is therefore
        hardcoded here.

        NOTE: By convention, at time step t, we are making predictions for y_{t+1} using x_t. So x_t can encode
        target values before or AT time t. As such, lags of 0 for input creation are perfectly valid for both
        input_lags and target_lags.

        For example: With the target of OT and inputs is [MUFL] with lags of [0, 1] for both input and target then the
        tensors look like
        Y = [OT_t,         OT_{t+1},  OT_{t+2},  ..., OT_n]^T

        X = [[OT_t,          OT_{t+1},       OT_{t+2},      ...,    OT_n],
             [OT_{t-1},      OT_t,           OT_{t+1},      ...,    OT_{n-1}],
             [MUFL_t,        MUFL_{t+1},     MUFL_{t+2},    ...,    MUFL_n],
             [MUFL_{t-1},    MUFL_t,         MUFL_{t+1},    ...,    MUFL_{n-1}]]^T

        In this setup, X[0, :] = [OT_t, OT_{t-1}, MUFL_t, MUFL_{t-1}] would be used to predict Y[1] = OT_{t+1}

        Args:
            inputs (List[InputFeatures]): These are the features in the dataset to use to help make predictions.
                If you want to include lagged values of the target in the input as well, set target_lag > 0.
            input_lags (List[int]): List of steps backward in input that should be included in the input features.
                For example, if input_lag is [1, 2] and we have MUFL and LULL features at inputs, then at time t, then
                x_t contains MUFL_{t-1}, MUFL_{t-2}, LULL_{t-1}, and LULL_{t-2}.
            target_lags (Optional[List[int]], optional): Similar to input lag, this is a list of steps backward in
                target values should be include in the input. For example, if target_lag is [1, 2], then at time t,
                x_t for y_t contains OT_{t-1} and OT_{t-2} in the input sequence  along with other input values.
                If none, then lagged targets are not included in the input.
            start_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to
                begin.  The minimum value for this argument is 2016-07-01 00:00:00. If None, the minimum value is
                used. If not the minimum value and input_lag/target_lag are greater than 1, we will still look back in
                time to gather these as far as possible. When prior time stamps are not available, we set the lagged
                values to 0. Defaults to None.
            end_date (Optional[datetime], optional): (INCLUSIVE) When in the dataset we want our time series to end.
                The maximum value for this argument is 2018-06-26 19:00:00. If None, the maximum value is
                used. Defaults to None.
            dtype (torch.dtype, optional): Default type for any torch tensors created. Defaults to torch.float64.
            dataset_path (str): Path to the dataset csv file. By default it is assumed to exist in the datasets/assets
                folder at "fedmoe/datasets/assets/ETTh1.csv".
        """
        assert (
            len(inputs) > 0 or target_lags is not None
        ), "No inputs specified. Either specify input features or specify a target lag"

        for input_lag in input_lags:
            assert input_lag >= 0, "Input lag must be at least 0"
        if target_lags is not None:
            for target_lag in target_lags:
                assert target_lag >= 0, "Target lag must be at least 0"

        self.inputs = inputs
        self.target = "OT"
        self.input_lags = input_lags
        self.target_lags = target_lags

        self.min_date = datetime(2016, 7, 1, 0)
        self.max_date = datetime(2018, 6, 26, 19)
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        assert start_date < end_date, "Start date occurs after end date. This is invalid"
        assert (
            self.min_date <= start_date <= self.max_date
        ), "Start date must occur on or after 2016-07-01 00:00:00 and on or before 2018-06-26 19:00:00"
        assert (
            self.min_date <= end_date <= self.max_date
        ), "End date must occur on or after 2016-07-01 00:00:00 and on or before 2018-06-26 19:00:00"

        self.start_date = start_date
        self.end_date = end_date
        self.dtype = dtype
        self.dataset_path = dataset_path

        self.raw_data = pd.read_csv(self.dataset_path)
        # Format the date column
        self.raw_data["date"] = pd.to_datetime(self.raw_data["date"])
        # Set date column as index
        self.raw_data = self.raw_data.set_index("date")

        # Get row indices of the first element and last element given the desired date ranges
        self.start_index, self.end_index = self._get_start_and_end_indices()

        self.total_time_steps = self.end_index - self.start_index + 1
        self.time_axis = torch.arange(0, self.total_time_steps)

        # Generate data set
        self.input_matrix, self.target_matrix = self._construct_dataset()

        self.x_dim = self.input_matrix.shape[1]
        self.y_dim = self.target_matrix.shape[1]

        del self.raw_data

    def _get_start_and_end_indices(self) -> Tuple[int, int]:
        start_index = self.raw_data.index.get_indexer([self.start_date], method="ffill")
        end_index = self.raw_data.index.get_indexer([self.end_date], method="ffill")
        return start_index[0], end_index[0]

    def _construct_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:

        input_tensors = []
        for lag in self.input_lags:
            # Filter the data to the time period we care about
            date_time_filtered = self._get_lagged_value(lag, self.raw_data)
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
                date_time_filtered = self._get_lagged_value(lag, self.raw_data)
                # Get the input values based on the input currencies
                target_df = date_time_filtered.loc[:, [self.target]]
                # Create tensor
                target_tensor = torch.from_numpy(target_df.values).to(self.dtype)
                # Left pad the tensor to the correct size with zeros in each row if it's too small
                n_rows, n_columns = target_tensor.shape
                pad_length = self.total_time_steps - n_rows
                input_tensors.append(torch.cat([torch.zeros((pad_length, n_columns)), target_tensor], dim=0))

        final_input_tensor = torch.cat(input_tensors, dim=1)

        # Note that the +1 here is because this slice is "non-inclusive" of the end_index
        date_time_filtered = self.raw_data[self.start_index : self.end_index + 1]
        target_df = date_time_filtered.loc[:, [self.target]]
        final_target_tensor = torch.from_numpy(target_df.values).to(self.dtype)

        return final_input_tensor, final_target_tensor

    def _get_lagged_value(self, lag: int, df: pd.DataFrame) -> pd.DataFrame:
        assert lag >= 0, "Lag cannot be negative"
        lagged_start_index = max(0, self.start_index - lag)
        lagged_end_index = self.end_index - lag
        # The +1 is because this indexing is non-inclusive
        return df[lagged_start_index : lagged_end_index + 1]

    def visualize(self) -> None:
        n_inputs = self.input_matrix.shape[1]
        n_targets = self.target_matrix.shape[1]

        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        for input_path in range(n_inputs):
            ax.plot(self.time_axis, self.input_matrix[:, input_path], linestyle="solid", linewidth=1.5)
        for target_path in range(n_targets):
            ax.plot(self.time_axis, self.target_matrix[:, target_path], linestyle="dotted", linewidth=1.5)
        ax.set_title("Features and Oil Temperature")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.show()

    def cut_first_time_steps(self, data_sequence_length: int, normalize: bool = False) -> None:
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
        if normalize:
            self.input_matrix = self.input_matrix[:data_sequence_length, :] / torch.max(self.input_matrix)
            self.target_matrix = self.target_matrix[:data_sequence_length, :] / torch.max(self.target_matrix)

    def maybe_random_cut_time_steps(
        self, data_sequence_length: int, start_index: int | None = None, normalize: bool = True
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
        # Normalize the data
        if normalize:
            selected_input_matrix = selected_input_matrix / torch.max(selected_input_matrix)
            selected_target_matrix = selected_target_matrix / torch.max(selected_target_matrix)
        return selected_input_matrix, selected_target_matrix

    def get_dataloader(self, num_samples: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        This function can be used to generate data loaders for train or validation, which are mainly
        used to pre-train the transformer model.
        Set the shuffle variable to True for validation data loader.
        """
        # Generate new data samples
        data: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        # Since the sampled sequence will be trimmed by one step later, we generate a longer sequence.
        for _ in range(num_samples):
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
