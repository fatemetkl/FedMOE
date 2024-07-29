import math

import torch
import torch.nn as nn

from fedmoe.clients.client import Client
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.tests.utils import get_data_and_target_sequences, get_transformer_client_manager

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()
Z_DIM = 5

def compute_objective(client: Client, beta: torch.Tensor, alpha: float, gamma: float, t: int) -> torch.Tensor:
    discount_2 = pow(math.e, -2.0 * alpha)
    discount_1 = pow(math.e, -1 * alpha)
    y_4 = TARGET_SEQUENCE[4].reshape(-1, 1)
    y_3 = TARGET_SEQUENCE[3].reshape(-1, 1)
    y_2 = TARGET_SEQUENCE[2].reshape(-1, 1)
    Y_3 = client.state.get_prediction_t(t - 2)
    Y_2 = client.state.get_prediction_t(t - 3)
    Y_1 = client.state.get_prediction_t(t - 4)
    Z_3 = client.state.get_hidden_state_t(t - 2).double()
    Z_2 = client.state.get_hidden_state_t(t - 3).double()
    Z_1 = client.state.get_hidden_state_t(t - 4).double()
    first_summand = torch.pow(torch.linalg.norm((y_4 - Y_3) - torch.matmul(Z_3, beta)), 2.0)
    second_summand = torch.pow(torch.linalg.norm((y_3 - Y_2) - torch.matmul(Z_2, beta)), 2.0)
    third_summand = torch.pow(torch.linalg.norm((y_2 - Y_1) - torch.matmul(Z_1, beta)), 2.0)
    regularizer = torch.pow(torch.linalg.norm(beta), 2.0)
    return first_summand + discount_1 * second_summand + discount_2 * third_summand + gamma * regularizer


def test_client_side_optimization() -> None:
    # Fixing seed for reproducible sampling trajectory
    torch.manual_seed(42)
    alpha = 1.5
    gamma = 2.0

    client_manager = get_transformer_client_manager(Z_DIM, sync_freq=3)

    # Making prediction for t=1
    t = 1
    # Temporarily bumping the time to make everything, will reset after.
    for client in client_manager.clients:
        client.state.next_time_step(t)
    # grab y_0
    last_observed_value = client_manager.get_y(t - 1)
    target_last_observed_value = torch.Tensor([0.3, 0.1 * 0.1 + 0.2, 0.7]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t - 1)
    X_t_target = client_0.state.Z_neg1
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)
    client_0_y_t = client_0.compute_y_t(t - 1)
    y_t_target = TARGET_SEQUENCE[t - 1].reshape(-1, 1) - client_0.state.Y_neg1
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(Z_DIM, dtype=torch.double)
    b = torch.matmul(X_t_target.T, y_t_target)
    client_0_beta_target = torch.linalg.solve(A.double(), b.double())
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t1 = client_0.feed_encoder(DATA_SEQUENCE[t - 1].reshape(-1, 1))
    client_0_preds_target_t1 = client_0.state.Y_0 + torch.matmul(client_0_hidden_state_t1.double(), client_0_beta_target)
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t1, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t - 1)
    client_1_y_t = client_1.compute_y_t(t - 1)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Need to put the time back one notch on all clients for this to work (because we bumped it above)
    for client in client_manager.clients:
        client.state._current_time -= 1
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[:, 0], client_0_preds_target_t1.squeeze())

    # Making prediction for t=2
    t = 2
    # Temporarily bumping the time to make everything, will reset after.
    for client in client_manager.clients:
        client.state.next_time_step(t)

    # grab y_1
    last_observed_value = client_manager.get_y(t - 1)
    target_last_observed_value = torch.Tensor([0.6, 0.2 * 0.2 + 0.4, 1.4]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t - 1)
    discount_1 = pow(math.e, -0.5 * alpha)
    X_t_target = torch.cat([discount_1 * client_0.state.Z_neg1, client_0_hidden_state_t1])
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)

    client_0_y_t = client_0.compute_y_t(t - 1)
    y_bar_first_block = discount_1 * (TARGET_SEQUENCE[t - 2].reshape(-1, 1) - client_0.state.Y_neg1)
    y_bar_second_block = TARGET_SEQUENCE[t - 1].reshape(-1, 1) - client_0.state.Y_0
    y_t_target = torch.cat([y_bar_first_block, y_bar_second_block])
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(Z_DIM, dtype=torch.double)
    b = torch.matmul(X_t_target.T.double(), y_t_target.double())
    client_0_beta_target = torch.linalg.solve(A.double(), b.double())
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t2 = client_0.feed_encoder(DATA_SEQUENCE[t - 1].reshape(-1, 1))
    # client_0_preds has the t=1 predictions from previous round
    client_0_preds_target_t2 = client_0_preds + torch.matmul(client_0_hidden_state_t2.double(), client_0_beta_target)
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t2, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t - 1)
    client_1_y_t = client_1.compute_y_t(t - 1)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Need to put the time back one notch on all clients for this to work (because we bumped it above)
    for client in client_manager.clients:
        client.state._current_time -= 1
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[:, 0], client_0_preds_target_t2.squeeze())

    # t = 5 (We want to make sure we're looking back the way we should )
    for t in [3, 4]:
        client_manager.fit_clients(t)
    # predicting for t=5
    t = 5
    # Temporarily bumping the time to make everything, will reset after.
    for client in client_manager.clients:
        client.state.next_time_step(t)
    # grab y_4
    last_observed_value = client_manager.get_y(t - 1)
    target_last_observed_value = torch.Tensor([1.5, 0.5 * 0.5 + 1.0, 3.5]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t - 1)
    discount_2 = pow(math.e, -1.0 * alpha)
    discount_1 = pow(math.e, -0.5 * alpha)
    X_t_target = torch.cat(
        [
            discount_2 * client_0.state.get_hidden_state_t(t - 4),
            discount_1 * client_0.state.get_hidden_state_t(t - 3),
            client_0.state.get_hidden_state_t(t - 2),
        ]
    )
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)

    client_0_y_t = client_0.compute_y_t(t - 1)
    y_t_target = torch.cat(
        [
            discount_2 * (TARGET_SEQUENCE[t - 3].reshape(-1, 1) - client_0.state.get_prediction_t(t - 4)),
            discount_1 * (TARGET_SEQUENCE[t - 2].reshape(-1, 1) - client_0.state.get_prediction_t(t - 3)),
            (TARGET_SEQUENCE[t - 1].reshape(-1, 1) - client_0.state.get_prediction_t(t - 2)),
        ]
    )
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(Z_DIM, dtype=torch.double)
    b = torch.matmul(X_t_target.T.double(), y_t_target.double())
    client_0_beta_target = torch.linalg.solve(A.double(), b.double())
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t5 = client_0.feed_encoder(DATA_SEQUENCE[t - 1].reshape(-1, 1))
    # client_0_preds has the t=1 predictions from previous round
    client_0_preds_target_t5 = client_0.state.get_prediction_t(t - 1) + torch.matmul(
        client_0_hidden_state_t5.double(), client_0_beta_target
    )
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t5, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t - 1)
    client_1_y_t = client_1.compute_y_t(t - 1)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Need to put the time back one notch on all clients for this to work (because we bumped it above)
    for client in client_manager.clients:
        client.state._current_time -= 1
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[:, 0], client_0_preds_target_t5.squeeze())

    # Ensure that beta is minimal for objective function
    opt_sum = compute_objective(client_0, client_0_beta, alpha, gamma, t)
    # test whether any randomly drawn betas are better
    for i in range(100000):
        test_beta = torch.randn((Z_DIM, 1)).double()
        test_sum = compute_objective(client_0, test_beta, alpha, gamma, t)
        assert test_sum > opt_sum, f"opt sum: {opt_sum}, test_sum: {test_sum}, test_beta: {test_beta}"
