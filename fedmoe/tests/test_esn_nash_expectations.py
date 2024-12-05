import torch

from fedmoe.game.echo_state_game import EchoStateGame
from fedmoe.tests.utils import get_data_and_target_sequences, get_esn_client_manager

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()


def test_esn_simulate_z_function() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, n_samples=5)

    # Need to fit the clients forward t=8 times to get the correct state and time.
    for t in range(0, 9):
        client_manager.fit_clients(t)

    client_0 = esn_game.clients[0]

    # Reset the manual seed here for the ESN encoders so we can follow the same basis generation in the manual
    # calculations below
    torch.manual_seed(42)
    esn_game.init_game_round_variables(8)
    # In sync_round of the server, we play the game within the 0, T time frame, so we use t=3 because that is
    # what is called on the server range(sync_freq - 1, -1 -1)
    Z_test = esn_game.simulate_z_t(3, client_0)

    # We are PREDICTING for t=9 and have a sync frequency of 4, so we start with Z_{3} and we need to
    # simulate forward using x_4, x_5, x_6, to get Z_4, Z_5, Z_6.
    # Let's do that manually.
    torch.manual_seed(42)
    # Reset the seed to be the same ESN basis generation as above.
    x_4 = DATA_SEQUENCE[4].reshape(-1, 1).double()
    input_4 = x_4.repeat(1, z_dim)
    Z_target = client_0.encoder(input_4, esn_game.get_z(game_t=-1, client=client_0), client_0.sigma)

    x_5 = DATA_SEQUENCE[5].reshape(-1, 1).double()
    input_5 = x_5.repeat(1, z_dim)
    Z_target = client_0.encoder(input_5, Z_target, client_0.sigma)

    x_6 = DATA_SEQUENCE[6].reshape(-1, 1).double()
    input_6 = x_6.repeat(1, z_dim)
    Z_target = client_0.encoder(input_6, Z_target, client_0.sigma)

    x_7 = DATA_SEQUENCE[7].reshape(-1, 1).double()
    input_7 = x_7.repeat(1, z_dim)
    Z_target = client_0.encoder(input_7, Z_target, client_0.sigma)

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
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, n_samples=3)

    for t in range(0, 8):
        client_manager.fit_clients(t)

    client_0 = esn_game.clients[0]

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN encoders so we can follow the same basis generation in the manual
    # calculations below
    esn_game.init_game_round_variables(7)
    # We use t=2 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    expectation_test = esn_game.get_expectation_e_zt(2, client_0)

    # We are PREDICTING for t=8 and have a sync frequency of 4, so we start with Z_{3} and we need to
    # simulate forward using x_4, x_5, x_6, to get Z_4, Z_5, Z_6.
    # Let's do that manually for the three samplings
    torch.manual_seed(42)
    # We use t=2 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    Z_6_1 = esn_game.simulate_z_t(2, client_0)
    Z_6_2 = esn_game.simulate_z_t(2, client_0)
    Z_6_3 = esn_game.simulate_z_t(2, client_0)

    e_0 = client_0.get_e(N)
    assert e_0.shape == (N * y_dim, y_dim)
    # Manually construct e_0
    e_0_target = torch.cat([torch.eye(y_dim), torch.zeros(y_dim, y_dim)], dim=0)
    torch.allclose(e_0, e_0_target.double(), rtol=0.0, atol=1e-8)
    e_0_Z_6_1 = torch.matmul(e_0.double(), Z_6_1.double())
    e_0_Z_6_2 = torch.matmul(e_0.double(), Z_6_2.double())
    e_0_Z_6_3 = torch.matmul(e_0.double(), Z_6_3.double())
    expectation_target = (1.0 / 3.0) * (e_0_Z_6_1 + e_0_Z_6_2 + e_0_Z_6_3)
    assert torch.allclose(expectation_target, expectation_test, rtol=0.0, atol=1e-5)


def test_esn_get_formation_of_a_ij_t_2() -> None:
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    y_dim = TARGET_SEQUENCE.shape[1]
    n_samples = 3
    game_t = 2

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    N = client_manager.num_clients
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, n_samples=n_samples)

    for t in range(0, 8):
        client_manager.fit_clients(t)

    client_0 = esn_game.clients[0]
    client_1 = esn_game.clients[1]

    esn_game.init_game_round_variables(7)
    # We need to patch in non zero P[T] values to make them non-trivial (they are initialized to zero)
    for client in client_manager.clients:
        client.P[game_t + 1] = torch.randn((N * y_dim, N * y_dim), dtype=torch.float64)
    assert isinstance(esn_game, EchoStateGame)

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN expectation calculations so we can follow the same basis generation in
    # the manual calculations below
    # We use t=2 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    A_01_8 = esn_game.get_A_ij_t(game_t, 0, 1)
    A_10_8 = esn_game.get_A_ij_t(game_t, 1, 0)
    A_00_8 = esn_game.get_A_ij_t(game_t, 0, 0)
    A_11_8 = esn_game.get_A_ij_t(game_t, 1, 1)

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN encoders so we can do our calculations
    # We use t=2 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    e_0_z_8 = esn_game.get_expectation_e_zt(game_t, client_0)
    e_1_z_8 = esn_game.get_expectation_e_zt(game_t, client_1)
    # Note that the only sync_freq Ps are stored, when computing
    # We consider P[3] because we reach for P[t+1]
    A_01_8_target = torch.matmul(torch.matmul(e_0_z_8.T, client_0.P[game_t + 1]), e_1_z_8)

    assert torch.allclose(A_01_8, A_01_8_target, rtol=0.0, atol=1e-5)

    # We use t=2 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    e_1_z_8 = esn_game.get_expectation_e_zt(game_t, client_1)
    e_0_z_8 = esn_game.get_expectation_e_zt(game_t, client_0)
    # We consider P[3] because we reach for P[t+1]
    A_10_8_target = torch.matmul(torch.matmul(e_1_z_8.T, client_1.P[game_t + 1]), e_0_z_8)

    assert torch.allclose(A_10_8, A_10_8_target, rtol=0.0, atol=1e-5)

    # Diagonal values are more complex
    # We'll simulate the trajectories out since they are correlated z values.
    samples = []
    for _ in range(n_samples):
        Z = esn_game.simulate_z_t(game_t, client_0)
        e_0 = torch.cat([torch.eye(y_dim), torch.zeros(y_dim, y_dim)], dim=0)
        e_z_T = torch.matmul(e_0.double(), Z.double())
        P_t_plus_1 = client_0.P[game_t + 1]
        sample = torch.matmul(torch.matmul(e_z_T.T, P_t_plus_1), e_z_T)
        samples.append(sample)
    sum_tensor = torch.zeros_like(samples[0])
    for sample in samples:
        sum_tensor = sum_tensor + sample
    A_00_8_target = sum_tensor / n_samples

    samples = []
    for _ in range(n_samples):
        Z = esn_game.simulate_z_t(game_t, client_1)
        e_1 = torch.cat([torch.zeros(y_dim, y_dim), torch.eye(y_dim)], dim=0)
        e_z_T = torch.matmul(e_1.double(), Z.double())
        P_t_plus_1 = client_1.P[game_t + 1]
        sample = torch.matmul(torch.matmul(e_z_T.T, P_t_plus_1), e_z_T)
        samples.append(sample)
    sum_tensor = torch.zeros_like(samples[0])
    for sample in samples:
        sum_tensor = sum_tensor + sample
    A_11_8_target = sum_tensor / n_samples

    assert torch.allclose(A_00_8, A_00_8_target, rtol=0.0, atol=1e-5)
    assert torch.allclose(A_11_8, A_11_8_target, rtol=0.0, atol=1e-5)


def test_esn_get_formation_of_a_ij_t_1() -> None:
    # NOTE: We tested the formation of the A blocks for game time = 2. Now we make sure everything works for a
    # different game time.
    alpha = 1.5
    gamma = 2.0
    sigma = torch.Tensor([[0.1], [0.2], [0.3]])
    z_dim = 3
    y_dim = TARGET_SEQUENCE.shape[1]
    n_samples = 3
    game_t = 1

    # Fixing seed for reproducible sampling trajectory etc.
    torch.manual_seed(42)

    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=4)
    N = client_manager.num_clients
    esn_game = EchoStateGame(client_manager.clients, 4, z_dim, n_samples=n_samples)

    for t in range(0, 8):
        client_manager.fit_clients(t)

    client_0 = esn_game.clients[0]
    client_1 = esn_game.clients[1]

    esn_game.init_game_round_variables(7)
    # We need to patch in non zero P[T-1] values to make them non-trivial (they are initialized to zero)
    for client in client_manager.clients:
        client.P[game_t + 1] = torch.randn((N * y_dim, N * y_dim), dtype=torch.float64)
    assert isinstance(esn_game, EchoStateGame)

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN expectation calculations so we can follow the same basis generation in
    # the manual calculations below
    # We use t=1 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    A_01_8 = esn_game.get_A_ij_t(game_t, 0, 1)
    A_10_8 = esn_game.get_A_ij_t(game_t, 1, 0)
    A_00_8 = esn_game.get_A_ij_t(game_t, 0, 0)
    A_11_8 = esn_game.get_A_ij_t(game_t, 1, 1)

    torch.manual_seed(42)
    # Reset the manual seed here for the ESN encoders so we can do our calculations
    # We use t=1 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    e_0_z_8 = esn_game.get_expectation_e_zt(game_t, client_0)
    e_1_z_8 = esn_game.get_expectation_e_zt(game_t, client_1)
    # Note that the only sync_freq Ps are stored, when computing
    # We consider P[2] because we reach for P[t+1]
    A_01_8_target = torch.matmul(torch.matmul(e_0_z_8.T, client_0.P[game_t + 1]), e_1_z_8)

    assert torch.allclose(A_01_8, A_01_8_target, rtol=0.0, atol=1e-5)

    # We use t=1 here to simulate the sync_freq - 2 that is done when calling sync_round from the server.
    e_1_z_8 = esn_game.get_expectation_e_zt(game_t, client_1)
    e_0_z_8 = esn_game.get_expectation_e_zt(game_t, client_0)
    # We consider P[2] because we reach for P[t+1]
    A_10_8_target = torch.matmul(torch.matmul(e_1_z_8.T, client_1.P[game_t + 1]), e_0_z_8)

    assert torch.allclose(A_10_8, A_10_8_target, rtol=0.0, atol=1e-5)

    # Diagonal values are more complex
    # We'll simulate the trajectories out since they are correlated z values.
    samples = []
    for _ in range(n_samples):
        Z = esn_game.simulate_z_t(game_t, client_0)
        e_0 = torch.cat([torch.eye(y_dim), torch.zeros(y_dim, y_dim)], dim=0)
        e_z_T = torch.matmul(e_0.double(), Z.double())
        P_t_plus_1 = client_0.P[game_t + 1]
        sample = torch.matmul(torch.matmul(e_z_T.T, P_t_plus_1), e_z_T)
        samples.append(sample)
    sum_tensor = torch.zeros_like(samples[0])
    for sample in samples:
        sum_tensor = sum_tensor + sample
    A_00_8_target = sum_tensor / n_samples

    samples = []
    for _ in range(n_samples):
        Z = esn_game.simulate_z_t(game_t, client_1)
        e_1 = torch.cat([torch.zeros(y_dim, y_dim), torch.eye(y_dim)], dim=0)
        e_z_T = torch.matmul(e_1.double(), Z.double())
        P_t_plus_1 = client_1.P[game_t + 1]
        sample = torch.matmul(torch.matmul(e_z_T.T, P_t_plus_1), e_z_T)
        samples.append(sample)
    sum_tensor = torch.zeros_like(samples[0])
    for sample in samples:
        sum_tensor = sum_tensor + sample
    A_11_8_target = sum_tensor / n_samples
    assert torch.allclose(A_00_8, A_00_8_target, rtol=0.0, atol=1e-5)
    assert torch.allclose(A_11_8, A_11_8_target, rtol=0.0, atol=1e-5)
