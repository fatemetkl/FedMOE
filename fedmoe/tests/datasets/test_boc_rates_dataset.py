from datetime import datetime

import torch
import pytest

from fedmoe.datasets.fedmoe_datasets.boc_rates import BankOfCanadaExchangeRates, ExchangeRates


def test_one_input_one_target() -> None:
    # Test one input currency, one target currency, but a lag of 1 time step for the target is also included
    # in the input.
    inputs = [ExchangeRates.AUD_CLOSE]
    targets = [ExchangeRates.USD_CLOSE]
    input_lag = 1
    target_lag = 1
    start_date = datetime(2007, 5, 10)
    end_date = datetime(2007, 5, 20)
    dataset = BankOfCanadaExchangeRates(
        inputs=inputs,
        targets=targets,
        input_lag=input_lag,
        target_lag=target_lag,
        start_date=start_date,
        end_date=end_date,
    )
    target_input_tensor = torch.tensor(
        [
            [0.9200, 1.1055],
            [0.9200, 1.1112],
            [0.9200, 1.1122],
            [0.9200, 1.1122],
            [0.9200, 1.1122],
            [0.9200, 1.1070],
            [0.9200, 1.0988],
            [0.9100, 1.1039],
            [0.9100, 1.0987],
            [0.9000, 1.0895],
            [0.9000, 1.0895],
        ]
    ).double()
    target_target_tensor = torch.tensor(
        [[1.1112], [1.1122], [1.1122], [1.1122], [1.1070], [1.0988], [1.1039], [1.0987], [1.0895], [1.0895], [1.0895]]
    ).double()
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)


def test_two_inputs_two_targets_with_lag() -> None:
    # Test two input currencies, two target currencies, two lagged steps for input and target are included in the
    # input as well.
    inputs = [ExchangeRates.AUD_CLOSE, ExchangeRates.DKK_CLOSE]
    targets = [ExchangeRates.GBP_CLOSE, ExchangeRates.USD_CLOSE]
    input_lag = 2
    target_lag = 2
    start_date = datetime(2007, 5, 10)
    end_date = datetime(2007, 5, 20)
    dataset = BankOfCanadaExchangeRates(
        inputs=inputs,
        targets=targets,
        input_lag=input_lag,
        target_lag=target_lag,
        start_date=start_date,
        end_date=end_date,
    )
    # Two target currencies and 11 steps
    assert dataset.target_matrix.shape == (11, 2)
    # Two target currencies with two steps of lag, two input currencies with two steps of lag (2*2 + 2*2)
    assert dataset.input_matrix.shape == (11, 8)
    target_input_tensor = torch.tensor(
        [
            [0.9200, 0.2000, 0.9200, 0.2000, 2.2038, 1.1055, 2.1976, 1.1047],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.1997, 1.1112, 2.2038, 1.1055],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.2041, 1.1122, 2.1997, 1.1112],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.2041, 1.1122, 2.2041, 1.1122],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.2041, 1.1122, 2.2041, 1.1122],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.1903, 1.1070, 2.2041, 1.1122],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.1812, 1.0988, 2.1903, 1.1070],
            [0.9100, 0.2000, 0.9200, 0.2000, 2.1820, 1.1039, 2.1812, 1.0988],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.1692, 1.0987, 2.1820, 1.1039],
            [0.9000, 0.2000, 0.9100, 0.2000, 2.1519, 1.0895, 2.1692, 1.0987],
            [0.9000, 0.2000, 0.9000, 0.2000, 2.1519, 1.0895, 2.1519, 1.0895],
        ],
    ).double()
    target_target_tensor = torch.tensor(
        [
            [2.1997, 1.1112],
            [2.2041, 1.1122],
            [2.2041, 1.1122],
            [2.2041, 1.1122],
            [2.1903, 1.1070],
            [2.1812, 1.0988],
            [2.1820, 1.1039],
            [2.1692, 1.0987],
            [2.1519, 1.0895],
            [2.1519, 1.0895],
            [2.1519, 1.0895],
        ]
    ).double()
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)


def test_two_inputs_two_targets_beyond_min_start() -> None:
    # Test two input currencies, two target currencies, two lagged steps for input and target are included in the
    # input as well, but both lag steps go beyond the start of the dataset. So we need to pad those steps properly.
    inputs = [ExchangeRates.AUD_CLOSE, ExchangeRates.DKK_CLOSE]
    targets = [ExchangeRates.GBP_CLOSE, ExchangeRates.USD_CLOSE]
    input_lag = 2
    target_lag = 2
    start_date = datetime(2007, 5, 1)
    end_date = datetime(2007, 5, 10)
    dataset = BankOfCanadaExchangeRates(
        inputs=inputs,
        targets=targets,
        input_lag=input_lag,
        target_lag=target_lag,
        start_date=start_date,
        end_date=end_date,
    )
    # Two target currencies and 11 steps
    assert dataset.target_matrix.shape == (10, 2)
    # Two target currencies with two steps of lag, two input currencies with two steps of lag (2*2 + 2*2)
    assert dataset.input_matrix.shape == (10, 8)
    target_input_tensor = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.9200, 0.2000, 0.0000, 0.0000, 2.2199, 1.1105, 0.0000, 0.0000],
            [0.9100, 0.2000, 0.9200, 0.2000, 2.2055, 1.1087, 2.2199, 1.1105],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.1999, 1.1066, 2.2055, 1.1087],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.2075, 1.1075, 2.1999, 1.1066],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.2075, 1.1075, 2.2075, 1.1075],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.2075, 1.1075, 2.2075, 1.1075],
            [0.9100, 0.2000, 0.9100, 0.2000, 2.1957, 1.1018, 2.2075, 1.1075],
            [0.9200, 0.2000, 0.9100, 0.2000, 2.1976, 1.1047, 2.1957, 1.1018],
            [0.9200, 0.2000, 0.9200, 0.2000, 2.2038, 1.1055, 2.1976, 1.1047],
        ]
    ).double()
    target_target_tensor = torch.tensor(
        [
            [2.2199, 1.1105],
            [2.2055, 1.1087],
            [2.1999, 1.1066],
            [2.2075, 1.1075],
            [2.2075, 1.1075],
            [2.2075, 1.1075],
            [2.1957, 1.1018],
            [2.1976, 1.1047],
            [2.2038, 1.1055],
            [2.1997, 1.1112],
        ]
    ).double()
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)

def test_two_inputs_no_target_lag() -> None:
    inputs = [ExchangeRates.AUD_CLOSE, ExchangeRates.DKK_CLOSE]
    targets = [ExchangeRates.GBP_CLOSE, ExchangeRates.USD_CLOSE]
    input_lag = 2
    start_date = datetime(2007, 5, 1)
    end_date = datetime(2007, 5, 10)
    dataset = BankOfCanadaExchangeRates(
        inputs=inputs,
        targets=targets,
        input_lag=input_lag,
        target_lag=None,
        start_date=start_date,
        end_date=end_date,
    )
    # Two target currencies and 11 steps
    assert dataset.target_matrix.shape == (10, 2)
    # Two target currencies with two steps of lag, two input currencies with two steps of lag (2*2 + 2*2)
    assert dataset.input_matrix.shape == (10, 4)
    target_input_tensor = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000,],
            [0.9200, 0.2000, 0.0000, 0.0000,],
            [0.9100, 0.2000, 0.9200, 0.2000,],
            [0.9100, 0.2000, 0.9100, 0.2000,],
            [0.9100, 0.2000, 0.9100, 0.2000,],
            [0.9100, 0.2000, 0.9100, 0.2000,],
            [0.9100, 0.2000, 0.9100, 0.2000,],
            [0.9100, 0.2000, 0.9100, 0.2000,],
            [0.9200, 0.2000, 0.9100, 0.2000,],
            [0.9200, 0.2000, 0.9200, 0.2000,],
        ]
    ).double()
    target_target_tensor = torch.tensor(
        [
            [2.2199, 1.1105],
            [2.2055, 1.1087],
            [2.1999, 1.1066],
            [2.2075, 1.1075],
            [2.2075, 1.1075],
            [2.2075, 1.1075],
            [2.1957, 1.1018],
            [2.1976, 1.1047],
            [2.2038, 1.1055],
            [2.1997, 1.1112],
        ]
    ).double()
    assert torch.allclose(dataset.input_matrix, target_input_tensor, rtol=0.0, atol=1e-6)
    assert torch.allclose(dataset.target_matrix, target_target_tensor, rtol=0.0, atol=1e-6)

def test_various_dataset_assertions() -> None:

    inputs = [ExchangeRates.AUD_CLOSE, ExchangeRates.DKK_CLOSE]
    targets = [ExchangeRates.GBP_CLOSE, ExchangeRates.USD_CLOSE]
    input_lag = 2
    target_lag = 2
    start_date = datetime(2007, 5, 3)
    end_date = datetime(2007, 5, 10)

    # inputs empty
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=[],
            targets=targets,
            input_lag=input_lag,
            target_lag=None,
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) == "No inputs specified. Either specify input features or specify a target lag"


    # input_lag <= 0
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lag=0,
            target_lag=target_lag,
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) == "Input lag must be at least 1"

    # target_lag <= 0
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lag=2,
            target_lag=0,
            start_date=start_date,
            end_date=end_date,
        )

    assert str(assertion_error.value) ==  "Target lag must be at least 1"

    # start date beyond minimum
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lag=input_lag,
            target_lag=target_lag,
            start_date=datetime(2007, 4, 3),
            end_date=end_date,
        )

    assert str(assertion_error.value) ==  "Start date must occur on or after 2007-05-01 and on or before 2017-04-28"

    # end date beyond maximum
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lag=input_lag,
            target_lag=target_lag,
            start_date=start_date,
            end_date=datetime(2024, 4, 3),
        )

    assert str(assertion_error.value) ==  "End date must occur on or after 2007-05-01 and on or before 2017-04-28"

    # end date before start date
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets,
            input_lag=input_lag,
            target_lag=target_lag,
            start_date=end_date,
            end_date=start_date,
        )

    assert str(assertion_error.value) ==  "Start date occurs after end date. This is invalid"

    # inputs and targets not mutually exclusive
    with pytest.raises(AssertionError) as assertion_error:
        BankOfCanadaExchangeRates(
            inputs=inputs,
            targets=targets + [ExchangeRates.AUD_CLOSE],
            input_lag=input_lag,
            target_lag=target_lag,
            start_date=end_date,
            end_date=start_date,
        )

    assert str(assertion_error.value) ==  "Inputs and targets should not overlap"