import torch
import torch.nn as nn

from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.models.echo_state_net import Esn
from fedmoe.tests.utils import get_data_and_target_sequences, get_esn_client_manager

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()


def test_esn_client_prediction_process() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3

    # Fixing seed for reproducible sampling trajectory
    torch.manual_seed(42)
    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim)

    # Making prediction for t=1
    t = 0
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_0 = torch.Tensor([[0.0346], [0.0184], [-0.0303]]).double()
    random_state_0 = torch.Tensor(
        [[-0.0636, -0.0685, -0.1286], [-0.1273, -0.1369, -0.2572], [-0.1909, -0.2054, -0.3857]]
    ).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_0 = client_0.state.get_hidden_state_t(t)

    beta_0_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_0, beta_0_target, rtol=0.0, atol=1e-3)
    x_0_matrix = DATA_SEQUENCE[0].reshape(-1, 1).repeat(1, z_dim).double()
    z_0_target = torch.matmul(client_0_encoder.A, x_0_matrix) + client_0_encoder.b + random_state_0
    z_0_target = z_0_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(-1).double())
    z_0_target = nn.Hardsigmoid()(z_0_target)

    assert torch.allclose(z_0_target, z_0, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_0_target, beta_0) + client_0.state.Y_0
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 1
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_1 = torch.Tensor([[0.2599], [0.1503], [0.1322]]).double()
    random_state_1 = torch.Tensor(
        [[0.0786, 0.1006, 0.0845], [0.1572, 0.2012, 0.1690], [0.2358, 0.3018, 0.2534]]
    ).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_1 = client_0.state.get_hidden_state_t(t)

    beta_1_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_1, beta_1_target, rtol=0.0, atol=1e-3)
    x_1_matrix = DATA_SEQUENCE[1].reshape(-1, 1).repeat(1, z_dim).double()
    z_1_target = torch.matmul(client_0_encoder.A, x_1_matrix) + client_0_encoder.b + random_state_1
    z_1_target = z_1_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(0).double())
    z_1_target = nn.Hardsigmoid()(z_1_target)

    assert torch.allclose(z_1_target, z_1, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_1_target, beta_1_target) + client_0.state.get_prediction_t(t)
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 2
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_2 = torch.Tensor([[0.3458], [0.2431], [0.2742]]).double()
    random_state_2 = torch.Tensor(
        [[0.1283, 0.0054, -0.0338], [0.2566, 0.0108, -0.0676], [0.3850, 0.0161, -0.1014]]
    ).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_2 = client_0.state.get_hidden_state_t(t)

    beta_2_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_2, beta_2_target, rtol=0.0, atol=1e-3)
    x_2_matrix = DATA_SEQUENCE[2].reshape(-1, 1).repeat(1, z_dim).double()
    z_2_target = torch.matmul(client_0_encoder.A, x_2_matrix) + client_0_encoder.b + random_state_2
    z_2_target = z_2_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(1).double())
    z_2_target = nn.Hardsigmoid()(z_2_target)

    assert torch.allclose(z_2_target, z_2, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_2_target, beta_2) + client_0.state.get_prediction_t(t)
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)
