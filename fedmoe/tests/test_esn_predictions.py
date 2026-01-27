import torch
from torch import nn

from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.models.echo_state_net import Esn
from fedmoe.tests.utils import get_data_and_target_sequences, get_esn_client_manager


DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()

torch.set_default_dtype(torch.float64)


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
    assert predictions.shape == (2, 3)

    beta_0 = torch.Tensor([[0.0179], [0.1740], [-0.0479]])
    random_state_0 = torch.Tensor(
        [
            [0.0547, 0.0643, -0.0790],
            [0.1094, 0.1286, -0.1581],
            [0.1641, 0.1929, -0.2371],
        ]
    )
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_0 = client_0.state.get_hidden_state_t(t)

    beta_0_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_0, beta_0_target, rtol=0.0, atol=1e-3)
    x_0_matrix = DATA_SEQUENCE[0].reshape(-1, 1).repeat(1, z_dim)
    z_0_target = torch.matmul(client_0_encoder.A, x_0_matrix) + client_0_encoder.b + random_state_0
    z_0_target = z_0_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(-1))
    z_0_target = nn.Hardsigmoid()(z_0_target)

    assert torch.allclose(z_0_target, z_0, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_0_target, beta_0) + client_0.state.Y_0
    assert torch.allclose(predictions[0, :].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 1
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (2, 3)

    beta_1 = torch.Tensor([[0.1372], [0.0927], [0.0534]])
    random_state_1 = torch.Tensor(
        [
            [0.2117, -0.1712, 0.0165],
            [0.4235, -0.3424, 0.0330],
            [0.6352, -0.5135, 0.0495],
        ]
    )
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_1 = client_0.state.get_hidden_state_t(t)

    beta_1_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_1, beta_1_target, rtol=0.0, atol=1e-3)
    x_1_matrix = DATA_SEQUENCE[1].reshape(-1, 1).repeat(1, z_dim)
    z_1_target = torch.matmul(client_0_encoder.A, x_1_matrix) + client_0_encoder.b + random_state_1
    z_1_target = z_1_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(0))
    z_1_target = nn.Hardsigmoid()(z_1_target)

    assert torch.allclose(z_1_target, z_1, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_1_target, beta_1_target) + client_0.state.get_prediction_t(t)
    assert torch.allclose(predictions[0, :].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 2
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (2, 3)

    beta_2 = torch.Tensor([[0.3037], [0.0661], [0.1090]])
    random_state_2 = torch.Tensor(
        [
            [0.1450, -0.0694, 0.0997],
            [0.2901, -0.1387, 0.1993],
            [0.4351, -0.2081, 0.2990],
        ]
    )
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, Esn)
    assert isinstance(client_0, EchoStateNetworkClient)
    z_2 = client_0.state.get_hidden_state_t(t)

    beta_2_target = client_0.state.get_beta_t(t)
    assert torch.allclose(beta_2, beta_2_target, rtol=0.0, atol=1e-3)
    x_2_matrix = DATA_SEQUENCE[2].reshape(-1, 1).repeat(1, z_dim)
    z_2_target = torch.matmul(client_0_encoder.A, x_2_matrix) + client_0_encoder.b + random_state_2
    z_2_target = z_2_target + torch.matmul(client_0_encoder.B, client_0.state.get_hidden_state_t(1))
    z_2_target = nn.Hardsigmoid()(z_2_target)

    assert torch.allclose(z_2_target, z_2, rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_2_target, beta_2) + client_0.state.get_prediction_t(t)
    assert torch.allclose(predictions[0, :].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)
