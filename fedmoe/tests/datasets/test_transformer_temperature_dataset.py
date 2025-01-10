from datetime import datetime

import pytest
import torch

from fedmoe.datasets.fedmoe_datasets.transformer_temperature import InputFeatures, TransformerTemperature

torch.set_default_dtype(torch.float64)


def test_zero_lag_for_target_and_input() -> None:
    # Recreating the dataset from the documentation of the transformer temperature dataset. OT target, MUFL input
    inputs = [InputFeatures.MUFL]
    input_lag = [0, 1]
    target_lag = [0, 1]
    start_date = datetime(2016, 7, 1, 13)
    end_date = datetime(2016, 7, 1, 23)
    dataset = TransformerTemperature(
        inputs=inputs,
        input_lags=input_lag,
        target_lags=target_lag,
        start_date=start_date,
        end_date=end_date,
    )

    # One target with 2 lags, one input with two lags
    assert dataset.input_matrix.shape == (11, 4)
    assert dataset.target_matrix.shape == (11, 1)

    target_input_tensor = torch.tensor(
        [
            [1.7770, 1.8120, 18.5720, 19.2050],
            [2.4520, 1.7770, 19.5560, 18.5720],
            [2.4870, 2.4520, 17.3050, 19.5560],
            [1.7060, 2.4870, 19.4860, 17.3050],
            [1.6350, 1.7060, 19.1340, 19.4860],
            [2.5230, 1.6350, 20.6820, 19.1340],
            [2.4520, 2.5230, 18.7120, 20.6820],
            [2.4520, 2.4520, 17.8680, 18.7120],
            [2.3810, 2.4520, 18.0090, 17.8680],
            [2.2030, 2.3810, 18.0090, 18.0090],
            [2.1320, 2.2030, 19.7680, 18.0090],
        ]
    )
    target_target_tensor = torch.tensor(
        [
            [18.5720],
            [19.5560],
            [17.3050],
            [19.4860],
            [19.1340],
            [20.6820],
            [18.7120],
            [17.8680],
            [18.0090],
            [18.0090],
            [19.7680],
        ]
    )
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)


def test_two_input_one_target_with_pad() -> None:
    # Test two feature inputs, but a lag of 1 time step for the target is also included in the input.
    # Because we're doing a lag from the beginning of the dataset, we also need to pad the start of each input
    # with a 0.0
    inputs = [InputFeatures.HUFL, InputFeatures.HULL]
    input_lag = [1]
    target_lag = [1]
    start_date = datetime(2016, 7, 1, 0)
    end_date = datetime(2016, 7, 1, 23)
    dataset = TransformerTemperature(
        inputs=inputs,
        input_lags=input_lag,
        target_lags=target_lag,
        start_date=start_date,
        end_date=end_date,
    )
    target_input_tensor = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [5.8270, 2.0090, 30.5310],
            [5.6930, 2.0760, 27.7870],
            [5.1570, 1.7410, 27.7870],
            [5.0900, 1.9420, 25.0440],
            [5.3580, 1.9420, 21.9480],
            [5.6260, 2.1430, 21.1740],
            [7.1670, 2.9470, 22.7920],
            [7.4350, 3.2820, 23.1440],
            [5.5590, 3.0140, 21.6670],
            [4.5550, 2.5450, 17.4460],
            [4.9570, 2.5450, 19.9790],
            [5.7600, 2.5450, 20.1190],
            [4.6890, 2.5450, 19.2050],
            [4.6890, 2.6790, 18.5720],
            [5.0900, 2.9470, 19.5560],
            [5.0900, 3.1480, 17.3050],
            [4.2200, 2.4110, 19.4860],
            [4.7560, 2.3440, 19.1340],
            [5.6260, 2.8800, 20.6820],
            [5.4920, 3.0140, 18.7120],
            [5.3580, 3.0140, 17.8680],
            [5.0900, 2.9470, 18.0090],
            [4.8230, 2.9470, 18.0090],
        ]
    )
    target_target_tensor = torch.tensor(
        [
            [30.5310],
            [27.7870],
            [27.7870],
            [25.0440],
            [21.9480],
            [21.1740],
            [22.7920],
            [23.1440],
            [21.6670],
            [17.4460],
            [19.9790],
            [20.1190],
            [19.2050],
            [18.5720],
            [19.5560],
            [17.3050],
            [19.4860],
            [19.1340],
            [20.6820],
            [18.7120],
            [17.8680],
            [18.0090],
            [18.0090],
            [19.7680],
        ]
    )
    assert dataset.input_matrix.shape == (24, 3)
    assert dataset.target_matrix.shape == (24, 1)
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)


def test_two_input_one_target_at_end() -> None:
    # Test two feature inputs, but a lag of 2 time steps for the target is also included in the input.
    # Because we're doing a lag from the beginning of the dataset, we also need to pad the start of each input
    # with a 0.0
    inputs = [InputFeatures.MUFL, InputFeatures.LULL]
    input_lag = [1, 2]
    target_lag = [1, 2]
    start_date = datetime(2018, 6, 26, 10)
    end_date = datetime(2018, 6, 26, 19)
    dataset = TransformerTemperature(
        inputs=inputs,
        input_lags=input_lag,
        target_lags=target_lag,
        start_date=start_date,
        end_date=end_date,
    )
    target_input_tensor = torch.tensor(
        [
            [-8.2090, 1.8580, -2.1680, 2.2840, 9.0750, 9.4260],
            [-10.4830, 1.6140, -8.2090, 1.8580, 8.9340, 9.0750],
            [-14.4630, 1.3400, -10.4830, 1.6140, 9.2150, 8.9340],
            [-17.5540, 0.8830, -14.4630, 1.3400, 9.2150, 9.2150],
            [-9.6660, 1.3100, -17.5540, 0.8830, 9.4260, 9.2150],
            [-4.9040, 1.4320, -9.6660, 1.3100, 10.2000, 9.4260],
            [-5.6150, 1.5230, -4.9040, 1.4320, 10.9040, 10.2000],
            [-9.1320, 1.6750, -5.6150, 1.5230, 11.0440, 10.9040],
            [-0.8170, 1.5230, -9.1320, 1.6750, 10.2710, 11.0440],
            [5.4720, 1.4320, -0.8170, 1.5230, 9.7780, 10.2710],
        ]
    )
    target_target_tensor = torch.tensor(
        [[8.9340], [9.2150], [9.2150], [9.4260], [10.2000], [10.9040], [11.0440], [10.2710], [9.7780], [9.5670]]
    )
    assert dataset.input_matrix.shape == (10, 6)
    assert dataset.target_matrix.shape == (10, 1)
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)


def test_various_dataset_assertions() -> None:

    inputs = [InputFeatures.HUFL, InputFeatures.HULL]
    input_lag = [1, 2]
    target_lag = [1, 2]
    start_date = datetime(2016, 7, 1, 0)
    end_date = datetime(2016, 7, 1, 23)

    # inputs empty
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=[],
            input_lags=input_lag,
            target_lags=None,
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) == "No inputs specified. Either specify input features or specify a target lag"

    # input_lag <= 0
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=inputs,
            input_lags=[-1],
            target_lags=target_lag,
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) == "Input lag must be at least 0"

    # target_lag <= 0
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=inputs,
            input_lags=[1, 2],
            target_lags=[-1],
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) == "Target lag must be at least 0"

    # start date beyond minimum
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=inputs,
            input_lags=input_lag,
            target_lags=target_lag,
            start_date=datetime(2007, 4, 3),
            end_date=end_date,
        )

    assert (
        str(assertion_error.value)
        == "Start date must occur on or after 2016-07-01 00:00:00 and on or before 2018-06-26 19:00:00"
    )

    # end date beyond maximum
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=inputs,
            input_lags=input_lag,
            target_lags=target_lag,
            start_date=start_date,
            end_date=datetime(2024, 4, 3),
        )

    assert (
        str(assertion_error.value)
        == "End date must occur on or after 2016-07-01 00:00:00 and on or before 2018-06-26 19:00:00"
    )

    # end date before start date
    with pytest.raises(AssertionError) as assertion_error:
        TransformerTemperature(
            inputs=inputs,
            input_lags=input_lag,
            target_lags=target_lag,
            start_date=end_date,
            end_date=start_date,
        )

    assert str(assertion_error.value) == "Start date occurs after end date. This is invalid"


def test_cutting_and_dataloader_functionalities() -> None:
    # Recreating the dataset from the documentation of the transformer temperature dataset. OT target, MUFL input
    inputs = [InputFeatures.MUFL]
    input_lag = [0, 1]
    target_lag = [0, 1]
    start_date = datetime(2016, 7, 1, 13)
    end_date = datetime(2016, 7, 1, 23)
    dataset = TransformerTemperature(
        inputs=inputs,
        input_lags=input_lag,
        target_lags=target_lag,
        start_date=start_date,
        end_date=end_date,
    )

    # One target with 2 lags, one input with two lags
    assert dataset.input_matrix.shape == (11, 4)
    assert dataset.target_matrix.shape == (11, 1)

    target_input_tensor = torch.tensor(
        [
            [1.7770, 1.8120, 18.5720, 19.2050],
            [2.4520, 1.7770, 19.5560, 18.5720],
            [2.4870, 2.4520, 17.3050, 19.5560],
            [1.7060, 2.4870, 19.4860, 17.3050],
            [1.6350, 1.7060, 19.1340, 19.4860],
            [2.5230, 1.6350, 20.6820, 19.1340],
            [2.4520, 2.5230, 18.7120, 20.6820],
            [2.4520, 2.4520, 17.8680, 18.7120],
            [2.3810, 2.4520, 18.0090, 17.8680],
            [2.2030, 2.3810, 18.0090, 18.0090],
            [2.1320, 2.2030, 19.7680, 18.0090],
        ]
    )
    target_target_tensor = torch.tensor(
        [
            [18.5720],
            [19.5560],
            [17.3050],
            [19.4860],
            [19.1340],
            [20.6820],
            [18.7120],
            [17.8680],
            [18.0090],
            [18.0090],
            [19.7680],
        ]
    )
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)
    # Now let's take the first n (n==5) elements of the data sample.
    dataset.cut_time_steps(5)
    assert dataset.input_matrix.shape == (5, 4)
    assert dataset.target_matrix.shape == (5, 1)
    # Make sure this is the first 5 elements of the original dataset.
    assert torch.allclose(dataset.input_matrix, target_input_tensor[:5, :], rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor[:5, :], rtol=0.0, atol=1e-6)

    # Now let's see if we can generate a dataloader from this dataset.
    dataloader = dataset.get_dataloader(num_samples=4, batch_size=2, shuffle=True)
    for input, target in dataloader:
        assert input.shape == (2, 5, 4)
        assert target.shape == (2, 5, 1)

    # Test maybe_random_cut_time_steps function
    non_random_input, non_random_output = dataset.maybe_random_cut_time_steps(data_sequence_length=4, start_index=3)
    target_non_random_input = torch.tensor(
        [
            [1.7060, 2.4870, 19.4860, 17.3050],
            [1.6350, 1.7060, 19.1340, 19.4860],
            [2.5230, 1.6350, 20.6820, 19.1340],
            [2.4520, 2.5230, 18.7120, 20.6820],
        ]
    )
    target_non_random_target = torch.tensor(
        [
            [19.4860],
            [19.1340],
            [20.6820],
            [18.7120],
        ]
    )
    assert torch.allclose(non_random_input, target_non_random_input, rtol=0.0, atol=1e-6)
    assert torch.allclose(non_random_output, target_non_random_target, rtol=0.0, atol=1e-6)
