import torch
import torch.nn as nn

from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.models.echo_state_net import Esn
from fedmoe.tests.utils import get_data_and_target_sequences, get_esn_client_manager
from fedmoe.game import EchoStateGame

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()


def test_esn_simulate_z_function() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    y_dim = TARGET_SEQUENCE.shape[1]

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)
    z_start = torch.randn((y_dim, z_dim))

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, N_samples=5)

    client_0 = esn_game.clients[0]

    # Reset the manual seed here for the ESN encoders so we can follow the same basis generation in the manual 
    # calculations below
    torch.manual_seed(42)
    Z_test = esn_game.simulate_z_t(8, client_0, z_start)

    # We are PREDICTING for t=8 we have a sync frequency of 4, so z_start is simulating Z_{2} and we need to 
    # simulate forward using x_3, x_4, x_5, x_6, to get Z_3, Z_4, Z_5, Z_6.
    # Let's do that manually. 
    torch.manual_seed(42)
    # Reset the seed to be the same ESN basis generation as above.
    x_3 = DATA_SEQUENCE[3].reshape(-1, 1).double()
    input_3 = x_3.repeat(1, z_dim)
    Z_target = client_0.encoder(input_3, z_start, client_0.sigma)

    x_4 = DATA_SEQUENCE[4].reshape(-1, 1).double()
    input_4 = x_4.repeat(1, z_dim)
    Z_target = client_0.encoder(input_4, Z_target, client_0.sigma)

    x_5 = DATA_SEQUENCE[5].reshape(-1, 1).double()
    input_5 = x_5.repeat(1, z_dim)
    Z_target = client_0.encoder(input_5, Z_target, client_0.sigma)

    x_6 = DATA_SEQUENCE[6].reshape(-1, 1).double()
    input_6 = x_6.repeat(1, z_dim)
    Z_target = client_0.encoder(input_6, Z_target, client_0.sigma)

    assert torch.allclose(Z_test, Z_target, rtol=0.0, atol=1e-5)

def test_esn_get_expectation_e_z_t() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    y_dim = TARGET_SEQUENCE.shape[1]

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    N = client_manager.num_clients
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, N_samples=3)

    for t in range(1, 9):
        client_manager.fit_clients(t)
    
    client_0 = esn_game.clients[0]

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN encoders so we can follow the same basis generation in the manual 
    # calculations below
    expectation_test = esn_game.get_expectation_e_zt(8, client_0)

    # We are PREDICTING for t=8 we have a sync frequency of 4, so z_start is simulating Z_{2} and we need to 
    # simulate forward using x_3, x_4, x_5, x_6, to get Z_3, Z_4, Z_5, Z_6.
    # Let's do that manually for the three samplings
    torch.manual_seed(42)

    Z_6_1 = esn_game.simulate_z_t(8 , client_0, client_0.state.get_hidden_state_t(2))
    Z_6_2 = esn_game.simulate_z_t(8 , client_0, client_0.state.get_hidden_state_t(2))
    Z_6_3 = esn_game.simulate_z_t(8 , client_0, client_0.state.get_hidden_state_t(2))

    e_0 = client_0.get_e(N)
    assert e_0.shape == (N*y_dim, y_dim)
    # Manually construct e_0
    e_0_target = torch.cat([torch.eye(y_dim), torch.zeros(y_dim, y_dim)], dim=0)
    torch.allclose(e_0, e_0_target.double(), rtol = 0.0, atol=1e-8)
    e_0_Z_6_1 = torch.matmul(e_0.double(), Z_6_1.double())
    e_0_Z_6_2 = torch.matmul(e_0.double(), Z_6_2.double())
    e_0_Z_6_3 = torch.matmul(e_0.double(), Z_6_3.double())
    expectation_target = (1.0/3.0) * (e_0_Z_6_1 + e_0_Z_6_2 + e_0_Z_6_3)
    assert torch.allclose(expectation_target, expectation_test, rtol=0.0, atol=1e-5)

def test_esn_get_formation_of_a_ij_t() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    y_dim = TARGET_SEQUENCE.shape[1]

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    N = client_manager.num_clients
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, N_samples=3)

    for t in range(1, 9):
        client_manager.fit_clients(t)

    client_0 = esn_game.clients[0]
    client_1 = esn_game.clients[1]

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN expectation calculations so we can follow the same basis generation in 
    # the manual calculations below
    assert isinstance(esn_game, EchoStateGame)
    A_01_8 = esn_game.get_A_ij_t(8, 0, 1)
    A_10_8 = esn_game.get_A_ij_t(8, 1, 0)
    A_00_8 = esn_game.get_A_ij_t(8, 0, 0)
    A_11_8 = esn_game.get_A_ij_t(8, 1, 1)

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN encoders so we can do our calculations
    e_0_z_8 = esn_game.get_expectation_e_zt(8, client_0)
    e_1_z_8 = esn_game.get_expectation_e_zt(8, client_1)
    # Note that the only sync_freq Ps are stored, when computing 
    A_01_8_target = torch.matmul(torch.matmul(e_0_z_8.T, client_0.P[2]), e_1_z_8)

    # Reset the manual seed here for the ESN encoders so we can do our calculations
    e_0_z_8 = esn_game.get_expectation_e_zt(8, client_0)
    e_1_z_8 = esn_game.get_expectation_e_zt(8, client_1)
    A_10_8_target = torch.matmul(torch.matmul(e_1_z_8.T, client_1.P[t + 1]), e_0_z_8)

    # Off diagonals first.
    assert torch.allclose(A_01_8, A_01_8_target, rtol=0.0, atol=1e-5)
    assert torch.allclose(A_01_8, A_01_8_target, rtol=0.0, atol=1e-5)

    # Diagonal values are more complex

