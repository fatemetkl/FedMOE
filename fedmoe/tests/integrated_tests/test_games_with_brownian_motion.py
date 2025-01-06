import torch

from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.datasets.brownian_motion_dataset import TimeSeriesBrownianTarget
from fedmoe.game.echo_state_game import EchoStateGame
from fedmoe.game.rfn_game import RfnGame
from fedmoe.game.transformer_game import TransformerGame
from fedmoe.server import Server
from fedmoe.tests.utils import get_transformer_client_manager, setup_transformer_structure_patch

torch.set_default_dtype(torch.float64)


def do_not_test_input_output_shapes_rfn() -> None:
    brownian_data_obj = TimeSeriesBrownianTarget(
        total_time_steps=100, n_brownian_trajectories=50, mu=1.0, sigma=2.0, offset=0.1
    )
    # In this example, the input_sequence is our target Brownian matrix
    input_sequence = brownian_data_obj.target_matrix
    assert input_sequence.shape == (100, 50)
    # Test the initial values to be set to our specified offset.
    assert torch.allclose(0.1 * torch.ones((50)), input_sequence[0, :], rtol=0.0, atol=1e-5)

    num_clients = 3
    T = 10
    hidden_dim = 4

    client_manager = ClientManager(
        ClientType.RFN,
        num_clients,
        input_sequence,
        T,
        hidden_dim,
        alpha=0.1,
        gamma=0.1,
        sigma=0.1,
    )
    game = RfnGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    # Run the server
    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=0.3,
        eta=4,
    )
    # RFN game is not implemented for dy>1 yet.
    _ = server.fit(num_rounds=98, have_sync=False)

    assert client_manager.get_y(t=10).shape == (50, 1)

    assert torch.cat(server.server_outputs, dim=1).shape == (50, 99)


def test_brownian_transformer(monkeypatch) -> None:
    brownian_data_obj = TimeSeriesBrownianTarget(
        total_time_steps=100, n_brownian_trajectories=50, mu=1.0, sigma=2.0, offset=0.1
    )
    # In this example, the input_sequence is our target Brownian matrix
    input_sequence = brownian_data_obj.target_matrix
    assert input_sequence.shape == (100, 50)
    # target_sequence is the first 25 dimension in Brownian matrix added to the last 25 dimensions for each time step.
    # Example: y(t) = x(t, 0:25) + x(t, 25:50)
    target_sequence = input_sequence[:, 0:25] + input_sequence[:, 25:50]

    assert target_sequence.shape == (100, 25)

    T = 10
    hidden_dim = 4
    # num_clients is 2
    monkeypatch.setattr(TransformerClient, "setup_transformer_structure", setup_transformer_structure_patch)
    client_manager = get_transformer_client_manager(
        z_dim=hidden_dim, sync_freq=T, data_sequence=input_sequence, target_sequence=target_sequence
    )

    game = TransformerGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    # Run the server
    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=0.3,
        eta=4,
    )

    # This tests passes even with game (have_sync=True).
    # This is set to False now to speed up the runtime of tests.
    _ = server.fit(num_rounds=98, have_sync=True)

    assert client_manager.get_y(t=10).shape == (25, 1)

    assert torch.cat(server.server_outputs, dim=1).shape == (25, 99)


def do_not_test_brownian_esn() -> None:
    brownian_data_obj = TimeSeriesBrownianTarget(
        total_time_steps=100, n_brownian_trajectories=50, mu=1.0, sigma=2.0, offset=0.1
    )
    # In this example, the input_sequence is our target Brownian matrix
    input_sequence = brownian_data_obj.target_matrix
    assert input_sequence.shape == (100, 50)
    # target_sequence is the first 25 dimension in input added to the last 25 dimensions for each time step.
    # Example: y(t) = x(t, 0:25) + x(t, 25:50)
    target_sequence = input_sequence[:, 0:25] + input_sequence[:, 25:50]

    assert target_sequence.shape == (100, 25)

    T = 10
    hidden_dim = 4
    num_clients = 3

    client_manager = ClientManager(
        ClientType.ESN,
        num_clients,
        input_sequence,
        T,
        hidden_dim,
        alpha=0.1,
        gamma=0.1,
        sigma=0.1,
        target_sequence=target_sequence,
    )

    game = EchoStateGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    # Run the server
    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=0.3,
        eta=4,
    )
    # This tests passes even with game (have_sync=True).
    # This is set to False now to speed up the runtime of tests.
    _ = server.fit(num_rounds=98, have_sync=False)

    assert client_manager.get_y(t=20).shape == (25, 1)

    assert torch.cat(server.server_outputs, dim=1).shape == (25, 99)
