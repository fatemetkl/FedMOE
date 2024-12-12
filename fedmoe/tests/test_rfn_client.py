import math

import torch

from fedmoe.clients.client import Client
from fedmoe.tests.utils import get_data_and_target_sequences, get_rfn_client_manager

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()


torch.set_default_dtype(torch.float64)


def compute_objective(client: Client, beta: torch.Tensor, alpha: float, gamma: float, t: int) -> torch.Tensor:
    discount_2 = pow(math.e, -2.0 * alpha)
    discount_1 = pow(math.e, -1 * alpha)
    y_4 = TARGET_SEQUENCE[4].reshape(-1, 1)
    y_3 = TARGET_SEQUENCE[3].reshape(-1, 1)
    y_2 = TARGET_SEQUENCE[2].reshape(-1, 1)
    Y_3 = client.state.get_prediction_t(t - 1)
    Y_2 = client.state.get_prediction_t(t - 2)
    Y_1 = client.state.get_prediction_t(t - 3)
    Z_3 = client.state.get_hidden_state_t(t - 1)
    Z_2 = client.state.get_hidden_state_t(t - 2)
    Z_1 = client.state.get_hidden_state_t(t - 3)
    first_summand = torch.pow(torch.linalg.norm((y_4 - Y_3) - torch.matmul(Z_3, beta)), 2.0)
    second_summand = torch.pow(torch.linalg.norm((y_3 - Y_2) - torch.matmul(Z_2, beta)), 2.0)
    third_summand = torch.pow(torch.linalg.norm((y_2 - Y_1) - torch.matmul(Z_1, beta)), 2.0)
    regularizer = torch.pow(torch.linalg.norm(beta), 2.0)
    return first_summand + discount_1 * second_summand + discount_2 * third_summand + gamma * regularizer


def test_client_side_optimization() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3

    client_manager = get_rfn_client_manager(alpha, gamma, sigma, z_dim, patch_client_state=True)

    # Making prediction for t=1
    t = 0
    # grab y_0
    last_observed_value = client_manager.get_y(t)
    target_last_observed_value = torch.Tensor([0.3, 0.1 * 0.1 + 0.2, 0.7]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t)
    X_t_target = client_0.state.Z_neg1
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)
    client_0_y_t = client_0.compute_y_t(t)
    y_t_target = TARGET_SEQUENCE[t].reshape(-1, 1) - client_0.state.Y_neg1
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(z_dim)
    b = torch.matmul(X_t_target.T, y_t_target)
    client_0_beta_target = torch.linalg.solve(A, b)
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t0 = client_0.feed_encoder(DATA_SEQUENCE[t].reshape(-1, 1))
    client_0_preds_target_t1 = client_0.state.Y_0 + torch.matmul(client_0_hidden_state_t0, client_0_beta_target)
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    # Incrementing the client 0's current time to 0 to be able to use predict(), will be reset later.
    client_0.state.next_time_step(next_time=t)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t1, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t)
    client_1_y_t = client_1.compute_y_t(t)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Resetting client 0's current time to -1 because it will be increased to 1 in fit_clients
    client_0.state._current_time = -1
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[0, :], client_0_preds_target_t1.squeeze())

    # Making prediction for t=2
    t = 1

    # grab y_1
    last_observed_value = client_manager.get_y(t)
    target_last_observed_value = torch.Tensor([0.6, 0.2 * 0.2 + 0.4, 1.4]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t)
    discount_1 = pow(math.e, -0.5 * alpha)
    X_t_target = torch.cat([discount_1 * client_0.state.Z_neg1, client_0_hidden_state_t0])
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)

    client_0_y_t = client_0.compute_y_t(t)
    y_bar_first_block = discount_1 * (TARGET_SEQUENCE[t - 1].reshape(-1, 1) - client_0.state.Y_neg1)
    y_bar_second_block = TARGET_SEQUENCE[t].reshape(-1, 1) - client_0.state.Y_0
    y_t_target = torch.cat([y_bar_first_block, y_bar_second_block])
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(z_dim)
    b = torch.matmul(X_t_target.T, y_t_target)
    client_0_beta_target = torch.linalg.solve(A, b)
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t1 = client_0.feed_encoder(DATA_SEQUENCE[t].reshape(-1, 1))
    # client_0_preds has the t=1 predictions from previous round
    client_0_preds_target_t2 = client_0_preds + torch.matmul(client_0_hidden_state_t1, client_0_beta_target)
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    # Incrementing the client 0's current time to 1 to be able to use predict(), will be reset later.
    client_0.state.next_time_step(next_time=t)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t2, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t)
    client_1_y_t = client_1.compute_y_t(t)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Resetting client 0's current time to 0 because it will be increased to 1 in fit_clients.
    client_0.state._current_time = 0
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[0, :], client_0_preds_target_t2.squeeze())

    # t = 5 (We want to make sure we're looking back the way we should )
    for t in [2, 3]:
        client_manager.fit_clients(t)
    # predicting for t=5
    t = 4

    # grab y_4
    last_observed_value = client_manager.get_y(t)
    target_last_observed_value = torch.Tensor([1.5, 0.5 * 0.5 + 1.0, 3.5]).reshape(-1, 1)
    assert torch.allclose(last_observed_value, target_last_observed_value, rtol=0.0, atol=1e-5)

    # Calculation for Client 0
    client_0 = client_manager.clients[0]
    client_0_X_t = client_0.compute_X_t(t)
    discount_2 = pow(math.e, -1.0 * alpha)
    discount_1 = pow(math.e, -0.5 * alpha)
    X_t_target = torch.cat(
        [
            discount_2 * client_0.state.get_hidden_state_t(t - 3),
            discount_1 * client_0.state.get_hidden_state_t(t - 2),
            client_0.state.get_hidden_state_t(t - 1),
        ]
    )
    assert torch.allclose(client_0_X_t, X_t_target, rtol=0.0, atol=1e-5)

    client_0_y_t = client_0.compute_y_t(t)
    y_t_target = torch.cat(
        [
            discount_2 * (TARGET_SEQUENCE[t - 2].reshape(-1, 1) - client_0.state.get_prediction_t(t - 3)),
            discount_1 * (TARGET_SEQUENCE[t - 1].reshape(-1, 1) - client_0.state.get_prediction_t(t - 2)),
            (TARGET_SEQUENCE[t].reshape(-1, 1) - client_0.state.get_prediction_t(t - 1)),
        ]
    )
    assert torch.allclose(client_0_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Manually perform ridge regression solution
    A = torch.matmul(X_t_target.T, X_t_target) + gamma * torch.eye(z_dim)
    b = torch.matmul(X_t_target.T, y_t_target)
    client_0_beta_target = torch.linalg.solve(A, b)
    client_0_beta = client_0.optimize_beta(t)
    assert torch.allclose(client_0_beta, client_0_beta_target, rtol=0.0, atol=1e-5)

    # Manually perform encoding
    # Set seed to freeze random state generation
    torch.manual_seed(42)
    client_0_hidden_state_t5 = client_0.feed_encoder(DATA_SEQUENCE[t].reshape(-1, 1))
    # client_0_preds has the t=1 predictions from previous round
    client_0_preds_target_t5 = client_0.state.get_prediction_t(t) + torch.matmul(
        client_0_hidden_state_t5, client_0_beta_target
    )
    # Set seed to reproduce random state generation from above.
    torch.manual_seed(42)
    # Incrementing the client 0's current time to 4 to be able to use predict(), will be reset later.
    client_0.state.next_time_step(next_time=t)
    _, _, client_0_preds = client_0.predict(t)
    assert torch.allclose(client_0_preds, client_0_preds_target_t5, rtol=0.0, atol=1e-5)

    # Calculations for Client 1 should be different
    client_1 = client_manager.clients[1]
    client_1_X_t = client_1.compute_X_t(t)
    client_1_y_t = client_1.compute_y_t(t)
    assert not torch.allclose(client_1_X_t, X_t_target, rtol=0.0, atol=1e-5)
    assert not torch.allclose(client_1_y_t, y_t_target, rtol=0.0, atol=1e-5)

    # Update Experts and return predictions
    torch.manual_seed(42)
    # Resetting client 0's current time to 3 because it will be increased to 4 in fit_clients.
    client_0.state._current_time = 3
    predictions = client_manager.fit_clients(t)
    assert torch.allclose(predictions[0, :], client_0_preds_target_t5.squeeze())

    # Ensure that beta is minimal for objective function
    opt_sum = compute_objective(client_0, client_0_beta, alpha, gamma, t)
    # test whether any randomly drawn betas are better
    for i in range(100000):
        test_beta = torch.randn((z_dim, 1))
        test_sum = compute_objective(client_0, test_beta, alpha, gamma, t)
        assert test_sum > opt_sum, f"opt sum: {opt_sum}, test_sum: {test_sum}, test_beta: {test_beta}"
