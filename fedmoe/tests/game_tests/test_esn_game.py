import torch

from fedmoe.game.echo_state_game import EchoStateGame
from fedmoe.metrics import MSEMetric
from fedmoe.server import Server
from fedmoe.tests.utils import get_esn_client_manager

torch.set_default_dtype(torch.float64)


def test_rfn_game_metric() -> None:
    torch.manual_seed(12)
    torch.set_default_dtype(torch.float64)
    z_dim = 3
    alpha = 0.01
    gamma = 2.1
    # Sigma should be (y_dim,1), and in this example y_dim is 3
    sigma = torch.Tensor([[0.01], [0.01], [0.01]])
    sync_freq = 2
    game_back_steps = 3
    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=game_back_steps)
    game = EchoStateGame(client_manager.clients, sync_freq=game_back_steps, z_dim=z_dim)
    server = Server(
        total_game_steps=game_back_steps,
        client_manager=client_manager,
        game=game,
        metrics=[MSEMetric("MSE")],
        game_freq=sync_freq,
        kappa=1.0,
        eta=1.0,
    )
    game_metric_value = server.fit(num_rounds=8, have_sync=True)

    # Restarting client states
    client_manager = get_esn_client_manager(alpha, gamma, sigma, z_dim, sync_freq=game_back_steps)
    game = EchoStateGame(client_manager.clients, sync_freq=game_back_steps, z_dim=z_dim)
    server = Server(
        total_game_steps=game_back_steps,
        client_manager=client_manager,
        game=game,
        metrics=[MSEMetric("MSE")],
        game_freq=sync_freq,
        kappa=1.0,
        eta=1.0,
    )
    non_game_metric_value = server.fit(num_rounds=8, have_sync=False)

    non_game_loss = non_game_metric_value["server - server_predictions - MSE"]
    game_loss = game_metric_value["server - server_predictions - MSE"]
    tolerance = 1e-5
    assert (
        game_loss < non_game_loss + tolerance
    ), f"Game loss: {game_loss} is greater than Non-game loss: {non_game_loss}"
