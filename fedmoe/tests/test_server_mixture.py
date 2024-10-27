import torch

from fedmoe.game import RfnGame
from fedmoe.metrics import RMSEMetric
from fedmoe.server import Server
from fedmoe.tests.utils import get_data_and_target_sequences, get_rfn_client_manager, get_rfn_client_manager_dy_dx_1


def compute_objective(
    w_star: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, kappa: float
) -> torch.Tensor:
    assert torch.allclose(torch.sum(w_star), torch.Tensor([1.0]).double(), rtol=0.0, atol=1e-5)
    residual = targets - torch.matmul(predictions.double(), w_star.double())
    residual_inner_product = torch.pow(torch.linalg.norm(residual), 2.0)
    regularizer = kappa * torch.pow(torch.linalg.norm(w_star), 2.0)
    return residual_inner_product + regularizer


def test_server_optimization() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    kappa = 0.5
    eta = 1
    N = 3

    client_manager = get_rfn_client_manager(alpha, gamma, sigma, z_dim, N)
    game = RfnGame(client_manager.clients, 3, z_dim)

    server = Server(sync_freq=3, client_manager=client_manager, game=game, metrics=[], kappa=kappa, eta=eta)

    predictions = torch.Tensor([[1.0, 0.5, 0.1], [2.0, 1.0, 0.2], [3.0, 1.5, 0.3]])
    y_t = torch.Tensor([0.75, 1.5, 2.0]).reshape(-1, 1)
    # Setup the mixture optimization pieces manually
    Y_tY_t = torch.Tensor(
        [
            [1 + 4 + 9, 0.5 + 2.0 + 3 * 1.5, 0.1 + 0.4 + 0.3 * 3.0],
            [0.5 + 2 + 3 * 1.5, 0.5 * 0.5 + 1 + 1.5 * 1.5, 0.5 * 0.1 + 0.2 + 0.3 * 1.5],
            [0.1 + 0.2 * 2.0 + 0.3 * 3.0, 0.1 * 0.5 + 0.2 + 0.3 * 1.5, 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3],
        ]
    )
    A = 2 * (Y_tY_t + kappa * torch.eye(N))
    y_tY_t = torch.Tensor(
        [[0.75 * 1 + 1.5 * 2 + 2.0 * 3], [0.75 * 0.5 + 1.5 * 1.0 + 2.0 * 1.5], [0.75 * 0.1 + 1.5 * 0.2 + 2.0 * 0.3]]
    )
    b = 2 * y_tY_t
    one_N = torch.ones(N).reshape(-1, 1)
    # Solve optimization problem manually
    A_inv = A.inverse()
    numerator = torch.matmul(one_N.T, torch.matmul(A_inv, b)) - eta
    denominator = torch.matmul(one_N.T, torch.matmul(A_inv, one_N))
    fraction = numerator / denominator
    rhs = b - fraction * one_N
    w_star_target = torch.matmul(A_inv, rhs).double()

    w_star = server.compute_mixture_weights(predictions, y_t).double()
    assert torch.allclose(w_star, w_star_target, rtol=0.0, atol=1e-6)

    # Test whether w_star is the minimizer
    opt_sum = compute_objective(w_star, predictions, y_t, kappa)
    # test whether any randomly drawn weights are better
    for i in range(100000):
        test_w = torch.randn((N, 1)).double()
        # Normalize to fit constraint
        test_w = eta * test_w / torch.sum(test_w)
        test_sum = compute_objective(test_w, predictions, y_t, kappa)
        assert test_sum > opt_sum, f"opt sum: {opt_sum}, test_sum: {test_sum}, test_w: {test_w}"


def test_server_mixture_weights_in_flow() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    kappa = 0.5
    eta = 1
    N = 3

    _, target_sequence = get_data_and_target_sequences()

    client_manager = get_rfn_client_manager(alpha, gamma, sigma, z_dim, N)
    game = RfnGame(client_manager.clients, 3, z_dim)

    server = Server(
        sync_freq=3, client_manager=client_manager, game=game, metrics=[RMSEMetric("RSME")], kappa=kappa, eta=eta
    )

    # Perform 5 rounds of client stepping and mixing without synchronization
    server.fit(num_rounds=5, have_sync=False, update_last_Y_sync=False)

    # Server should have seen y_t values for t=0 to 4
    assert len(server.observed_values) == 5
    observed_values_target = [target_sequence[i] for i in range(0, 5)]
    for observed_value_target, observed_value in zip(observed_values_target, server.observed_values):
        torch.allclose(observed_value_target, observed_value, rtol=0.0, atol=1e-6)

    # Server should have mixture weights calculated for t=0, 1, ..., 4
    assert len(server.mixture_weights) == 5
    assert torch.allclose(
        server.mixture_weights[1], torch.Tensor([[0.0773], [0.0724], [0.8503]]).double(), rtol=0.0, atol=1e-3
    )
    assert torch.allclose(
        server.mixture_weights[4], torch.Tensor([[-0.5346], [0.6333], [0.9014]]).double(), rtol=0.0, atol=1e-3
    )

    # Clients should have been asked to provide predictions for t=1,..., 5, along with the a priori initialized
    # \hat{Y}_0^i
    assert len(server.clients_predictions) == 6
    for client_preds in server.clients_predictions:
        assert client_preds.shape == (server.y_dim, server.num_clients)

    # Server should have made predictions for t=1, ..., t=5
    assert len(server.server_outputs) == 6
    for server_preds in server.server_outputs:
        assert server_preds.shape == (server.y_dim, 1)

    # Make sure the relationship between the client predictions and server weights is correctly reflected in the
    # server outputs
    # Server output for time t+1 is the product of clients' output at time t+1 and mixture weights at t,
    # so the first elements in the below tests are `server_output[1] = client_predictions[1] * mixture_weights[0]`.
    for client_preds, weights, server_output in zip(
        server.clients_predictions[1:], server.mixture_weights, server.server_outputs[1:]
    ):
        assert torch.allclose(
            torch.matmul(client_preds.double(), weights.double()), server_output.double(), rtol=0.0, atol=1e-6
        )


def test_full_flow_with_dy_dx_one() -> None:
    alpha = 1.5
    gamma = 2.0
    z_dim = 3
    kappa = 0.5
    eta = 1
    N = 3

    client_manager = get_rfn_client_manager_dy_dx_1(alpha, gamma, z_dim, N)
    game = RfnGame(client_manager.clients, 3, z_dim)

    server = Server(
        sync_freq=3, client_manager=client_manager, game=game, metrics=[RMSEMetric("RSME")], kappa=kappa, eta=eta
    )

    # Perform 5 rounds of client stepping and mixing without synchronization, just making sure it runs all the way
    # through
    server.fit(num_rounds=5, have_sync=False, update_last_Y_sync=False)
