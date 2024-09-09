import torch

from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.datasets.brownian_motion_dataset import get_brownian_data_sequences
from fedmoe.game import RfnGame
from fedmoe.server import Server


def test_input_output_shapes() -> None:
    input_sequence = get_brownian_data_sequences(
        n_brownian_trajectories=50, time_steps=100, mu=1.0, sigma=2.0, offset=0.1
    )
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
        sync_freq=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=0.3,
        eta=4,
    )
    _ = server.fit(num_rounds=98, have_sync=False)

    random_time = 10
    assert client_manager.get_y(random_time).shape == (50, 1)

    assert torch.cat(server.server_outputs, dim=1).shape == (50, 99)
