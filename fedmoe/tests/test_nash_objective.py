import torch

from fedmoe.game.rfn_game import RfnGame
from fedmoe.metrics import RMSEMetric
from fedmoe.server import Server
from fedmoe.tests.test_game_utils import compute_game_regret_objective
from fedmoe.tests.utils import get_rfn_client_manager_dy_dx_1


def do_not_test_nash_game_objective_sync_step() -> None:
    """
    Testing if playing the Nash game reduces the regret value at each synchronization step.
    """
    # Assuming that the code is correct based on the paper, let's examine the solution.
    data_length = 20
    sync_freq = 3
    client_manager = get_rfn_client_manager_dy_dx_1(
        alpha=0.1,
        gamma=0.1,
        z_dim=3,
        sigma=torch.Tensor([[0.1]]),
        data_length=data_length,
        sync_freq=sync_freq,
        patch_client_state=True,
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
    # Important: don't forget to pass the new clients to the game also!
    client_manager.clients = client_manager.initiate_clients()
    game.clients = client_manager.clients
    simple_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = simple_server.fit(data_length - 1, have_sync=False, update_last_Y_sync=False)

    # Reset clients
    # Important: don't forget to pass the new clients to the game also!
    client_manager.clients = client_manager.initiate_clients()
    game.clients = client_manager.clients
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
                    1,
                )
                game_no_Y_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - 1)],
                    [target[i]],
                    [game_server_no_Y.server_outputs[i]],
                    client_manager.gamma,
                    client_manager.alpha,
                    1,
                )
                no_game_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - 1)],
                    [target[i]],
                    [simple_server.server_outputs[i]],
                    client_manager.gamma,
                    client_manager.alpha,
                    1,
                )
                assert (
                    game_no_Y_regret < no_game_regret
                ), f"Failed at index {i} client {client_id}\
                    no_game_regret: {no_game_regret}, game_no_Y_regret: {game_no_Y_regret}"


def do_not_test_nash_game_objective_accumulative() -> None:
    """
    Testing if playing the Nash game reduces the regret value throughout the whole prediction length.
    """
    data_length = 20
    sync_freq = 3
    client_manager = get_rfn_client_manager_dy_dx_1(
        alpha=0.1,
        gamma=0.1,
        z_dim=3,
        sigma=torch.Tensor([[0.1]]),
        data_length=data_length,
        sync_freq=sync_freq,
        patch_client_state=True,
    )
    target = client_manager.common_target_sequence
    game = RfnGame(
        client_manager.clients,
        sync_freq=sync_freq,
        z_dim=3,
    )

    simple_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = simple_server.fit(data_length - 1, have_sync=False, update_last_Y_sync=False)

    # Reset clients for the new experiment
    # Important: don't forget to pass the new clients to the game also!
    client_manager.clients = client_manager.initiate_clients()
    game.clients = client_manager.clients
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

                game_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - j - 1) for j in range(0, game.sync_freq)],
                    [target[i - j] for j in range(0, game.sync_freq)],
                    [game_server_no_Y.server_outputs[i - j] for j in range(0, game.sync_freq)],
                    client_manager.gamma,
                    client_manager.alpha,
                    backward_time_length=sync_freq,
                )
                game_sum_regret += int(game_regret)

                no_game_regret = compute_game_regret_objective(
                    [client.state.get_beta_t(i - j - 1) for j in range(0, game.sync_freq)],
                    [target[i - j] for j in range(0, game.sync_freq)],
                    [simple_server.server_outputs[i - j] for j in range(0, game.sync_freq)],
                    client_manager.gamma,
                    client_manager.alpha,
                    backward_time_length=sync_freq,
                )
                simple_sum_regret += int(no_game_regret)

    assert (
        game_sum_regret < simple_sum_regret
    ), f" Accumulative game_regret: {game_sum_regret}, no_game_regret: {simple_sum_regret}"


def do_not_test_nash_beta() -> None:
    seed = 2024
    torch.manual_seed(seed)
    data_length = 20
    sync_freq = 5
    client_manager = get_rfn_client_manager_dy_dx_1(
        alpha=0.1,
        gamma=0.1,
        z_dim=3,
        sigma=torch.tensor([[0.1]]),
        data_length=data_length,
        sync_freq=sync_freq,
        patch_client_state=True,
    )
    target = client_manager.common_target_sequence
    game = RfnGame(
        client_manager.clients,
        sync_freq=sync_freq,
        z_dim=3,
    )

    game_server = Server(sync_freq, client_manager, game, metrics=[RMSEMetric("RSME")], kappa=2.0, eta=1)
    _ = game_server.fit(data_length - 1, have_sync=True, update_last_Y_sync=False)
    clients = client_manager.clients

    # We have to test it for both clients because game works on the mixture weights (so we need all clients).
    for i in range(0, len(target)):
        if i % game.sync_freq == 0 and i > 0:
            # Check if the Nash beta is minimal based on the regret function
            # Get the Nash beta
            opt_regret_0 = compute_game_regret_objective(
                [clients[0].state.get_beta_t(i - 1)],
                [target[i]],
                [game_server.server_outputs[i]],
                client_manager.gamma,
                client_manager.alpha,
                1,
            )
            opt_regret_1 = compute_game_regret_objective(
                [clients[1].state.get_beta_t(i - 1)],
                [target[i]],
                [game_server.server_outputs[i]],
                client_manager.gamma,
                client_manager.alpha,
                1,
            )
            opt_regret = opt_regret_0 + opt_regret_1
            # test whether any randomly drawn betas are better
            for j in range(100000):
                # z_dim = 3
                # beta shape is N x d_z x d_y
                test_betas = torch.randn((2, 3, 1)).double()
                # New predictions shape is d_y x 1
                new_preds = client_manager.get_predictions_with_beta(i, test_betas)
                # Nash betas are replaced with random beta in each client
                # Compute new server output based on old mixture weights
                new_server_output = torch.matmul(new_preds.double(), game_server.mixture_weights[i].double())
                test_regret_0 = compute_game_regret_objective(
                    [test_betas[0]],
                    [target[i]],
                    [new_server_output],
                    client_manager.gamma,
                    client_manager.alpha,
                    1,
                )
                test_regret_1 = compute_game_regret_objective(
                    [test_betas[1]],
                    [target[i]],
                    [new_server_output],
                    client_manager.gamma,
                    client_manager.alpha,
                    1,
                )

                test_regret = test_regret_1 + test_regret_0
                assert (
                    test_regret > opt_regret
                ), f" opt regret: {opt_regret}, test regret: {test_regret}, test_beta: {test_betas},\
                      target: {target[i]}, game pred: {game_server.server_outputs[i]},\
                            random pred: {new_server_output}"
