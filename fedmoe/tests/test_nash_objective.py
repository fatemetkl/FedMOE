from fedmoe.game import RfnGame
from fedmoe.metrics import RMSEMetric
from fedmoe.server import Server
from fedmoe.tests.game_utils import compute_game_regret_objective
from fedmoe.tests.utils import get_client_manager_dy_dx_1


def test_nash_game_objective_sync_step() -> None:
    """
    Testing if playing the Nash game reduces the regret value at each synchronization step.
    """
    # Assuming that the code is correct based on the paper, let's examine the solution.
    data_length = 20
    sync_freq = 3
    client_manager = get_client_manager_dy_dx_1(
        alpha=0.1, gamma=0.1, z_dim=3, data_length=data_length, sync_freq=sync_freq
    )
    target = client_manager.common_target_sequence
    game = RfnGame(
        client_manager.clients,
        sync_freq=sync_freq,
        z_dim=3,
    )
    game_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = game_server.fit(data_length - 1, have_sync=True, update_last_Y_sync=True)

    # Reset clients
    client_manager.clients = client_manager.initiate_clients()
    simple_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = simple_server.fit(data_length - 1, have_sync=False, update_last_Y_sync=False)

    client_manager.clients = client_manager.initiate_clients()
    game_server_no_Y = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = game_server_no_Y.fit(data_length - 1, have_sync=True, update_last_Y_sync=False)

    assert len(target) == len(game_server.server_outputs) == len(simple_server.server_outputs)

    # Comparing the regret functions in different scenarios to see if game minimizes the regret.
    for i in range(0, len(target)):
        if i % game.sync_freq == 0 and i > 0:
            # print(target[i])
            for client_id in range(0, client_manager.num_clients):
                client = client_manager.clients[client_id]

                _ = compute_game_regret_objective(
                    [client.state.get_beta_t(i - 1)],
                    [target[i]],
                    [game_server.server_outputs[i]],
                    client_manager.gamma,
                    client_manager.alpha,
                    0,
                )
                game_no_Y_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - 1)],
                    [target[i]],
                    [game_server_no_Y.server_outputs[i]],
                    client_manager.gamma,
                    client_manager.alpha,
                    0,
                )
                no_game_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - 1)],
                    [target[i]],
                    [simple_server.server_outputs[i]],
                    client_manager.gamma,
                    client_manager.alpha,
                    0,
                )
                assert (
                    game_no_Y_regret < no_game_regret
                ), f"Failed at index {i} client {client_id}\
                    no_game_regret: {no_game_regret}, game_no_Y_regret: {game_no_Y_regret}"


def test_nash_game_objective_accumulative() -> None:
    """
    Testing if playing the Nash game reduces the regret value throughout the whole prediction length.
    """
    data_length = 20
    sync_freq = 3
    client_manager = get_client_manager_dy_dx_1(
        alpha=0.1, gamma=0.1, z_dim=3, data_length=data_length, sync_freq=sync_freq
    )
    target = client_manager.common_target_sequence
    game = RfnGame(
        client_manager.clients,
        sync_freq=sync_freq,
        z_dim=3,
    )
    # Reset clients
    client_manager.clients = client_manager.initiate_clients()
    simple_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = simple_server.fit(data_length - 1, have_sync=False, update_last_Y_sync=False)

    client_manager.clients = client_manager.initiate_clients()
    game_server_no_Y = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = game_server_no_Y.fit(data_length - 1, have_sync=True, update_last_Y_sync=False)

    assert len(target) == len(simple_server.server_outputs)

    # Comparing the regret functions in different scenarios to see if game minimizes the regret.
    game_sum_regret = 0
    simple_sum_regret = 0
    for i in range(0, len(target)):
        if i % game.sync_freq == 0 and i > 0:
            for client_id in range(0, client_manager.num_clients):
                client = client_manager.clients[client_id]

                game_no_Y_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(j) for j in range(i - sync_freq, i)],
                    [target[j] for j in range(i - sync_freq + 1, i + 1)],
                    [game_server_no_Y.server_outputs[j] for j in range(i - sync_freq + 1, i + 1)],
                    client_manager.gamma,
                    client_manager.alpha,
                    backward_time_length=(sync_freq - 1),
                )
                game_sum_regret += game_no_Y_regret

                no_game_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(j) for j in range(i - sync_freq, i)],
                    [target[j] for j in range(i - sync_freq + 1, i + 1)],
                    [simple_server.server_outputs[j] for j in range(i - sync_freq + 1, i + 1)],
                    client_manager.gamma,
                    client_manager.alpha,
                    backward_time_length=(sync_freq - 1),
                )
                simple_sum_regret += no_game_regret

    assert (
        game_sum_regret < simple_sum_regret
    ), f" Accumulative no_game_regret: {game_sum_regret}, game_no_Y_regret: {simple_sum_regret}"
