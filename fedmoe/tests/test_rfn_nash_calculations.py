# type: ignore
import random
from typing import Tuple

import torch

from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.game.rfn_game import RfnGame
from fedmoe.server import Server
from fedmoe.tests.manual_calculations import (
    calculate_B1,
    calculate_C1,
    calculate_D1,
    calculate_p1_manually,
    calculate_st_manually,
    calculate_z1_manually,
    compute_block_A_00,
    compute_block_A_01,
    manual_block_1,
)
from fedmoe.tests.test_game_utils import compute_game_regret_objective

torch.set_default_dtype(torch.float64)


class experiment_setup:
    def __init__(self, y_dim: int, z_dim: int, sync_freq: int, alpha: float, gamma: float, sigma: float) -> None:
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.sync_freq = sync_freq
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.eta = 2


def set_data_target() -> Tuple[torch.Tensor, torch.Tensor]:
    # F(x) = 2x
    data = torch.Tensor([[1], [2], [3], [4], [5]])
    target = torch.Tensor([[2], [4], [6], [8], [10]])
    return data, target


def set_data_target_long() -> Tuple[torch.Tensor, torch.Tensor]:
    # F(x) = 2x
    data = torch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    target = torch.Tensor([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]])
    return data, target


def _do_not_test_server_game() -> None:
    seed = 2024
    random.seed(seed)
    torch.manual_seed(seed)
    T = 3
    z_dim = 2
    num_clients = 2
    data, target = set_data_target()
    exp_var = experiment_setup(1, z_dim, T, 0.1, 0.1, 0.001)

    client_manager = ClientManager(
        ClientType.RFN, num_clients, data, T, exp_var.z_dim, exp_var.alpha, exp_var.gamma, exp_var.sigma, target
    )
    game = RfnGame(
        client_manager.clients,
        sync_freq=exp_var.sync_freq,
        z_dim=exp_var.z_dim,
    )

    server = Server(T, client_manager, game, metrics=[], kappa=2.0, eta=1)
    #  Compute mixture weights for t = 0, 1 ,2
    t = 0

    w_0 = torch.randn((num_clients, 1))
    w_0 = exp_var.eta * w_0 / torch.sum(w_0)
    predictions_0 = client_manager.get_Y_0()

    # Prediction for t = 1
    t = 0
    y_0 = target[0].reshape(exp_var.y_dim, 1)
    predictions_1 = client_manager.fit_clients(t)
    w_1 = server.compute_mixture_weights(predictions_0, y_0)
    # server_pred_1 = torch.matmul(predictions_1, w_1)

    # Predictions for t = 2
    t = 1
    y_1 = target[1].reshape(exp_var.y_dim, 1)
    predictions_2 = client_manager.fit_clients(t)
    w_2 = server.compute_mixture_weights(predictions_1, y_1)
    # server_pred_2 = torch.matmul(predictions_2, w_2)

    # Now sync step T = 3
    t = 2
    y_2 = target[2].reshape(exp_var.y_dim, 1)
    predictions_3 = client_manager.fit_clients(t)
    w_3 = server.compute_mixture_weights(predictions_2, y_2)

    # Automatic game calculations
    past_T_betas_from_game = server.sync_round(
        t,
        [y_0, y_1, y_2],
        [w_0, w_1, w_2],
    )
    #  Manual game
    #  First initiate P(T-1) and S(T-1) --> P(2) and S(2)
    game.init_game_round_variables(current_time=3)

    # Step 1) initiate P and S for T-1 in clients (2 = T-1)
    game.first_block_alg2(w_2, y_2, time=2)
    manual_P_T, manual_S_T = manual_block_1()

    assert torch.allclose(game.clients[0].P[2], torch.Tensor(manual_P_T), rtol=0.0, atol=1e-5)
    assert torch.allclose(game.clients[0].S[2], torch.Tensor(manual_S_T), rtol=0.0, atol=1e-5)

    #  Go over all the steps before T
    #  Test the backward loop (t = T-2, ..., 0 do)
    t = 1
    #  \bold{w}_t
    assert w_1.shape == (num_clients, exp_var.y_dim)

    bold_w_t_1 = game.create_bold_w_t(w_1)

    assert bold_w_t_1.shape == (num_clients * exp_var.y_dim, exp_var.y_dim)
    #  test A component
    # TODO: calculate these values by hand to confirm
    A_1 = game.calculate_a(time=1)
    game.set_A_t(t, A_1)

    # Every manual test is for client 1 (game.clients[0])
    manual_Z_client_0, a_client_0 = calculate_z1_manually(game.clients)
    game_Z_client_0 = game.get_expectation_zt(time=1, client=game.clients[0])
    # Check that calculated a(t, i) is correct
    assert torch.allclose(manual_Z_client_0, game_Z_client_0, rtol=0.0, atol=1e-4)

    game_Z_client_1 = game.get_expectation_zt(time=1, client=game.clients[1])

    # Test A block calculation
    # A for i!=j (i = 0, j = 1)
    manual_block_A_01 = compute_block_A_01(P_next=manual_P_T)
    game_block_A_01 = game.get_A_ij_t(1, 0, 1)
    assert torch.allclose(manual_block_A_01, game_block_A_01, rtol=0.0, atol=1e-4)

    # A for i == j == 0 (i is client id 0)
    manual_block_A_00 = compute_block_A_00()
    game_block_A_00 = game.get_A_ij_t(1, 0, 0)
    assert torch.allclose(manual_block_A_00, game_block_A_00, rtol=0.0, atol=1e-4)

    B_1 = game.calculate_b(t=1)
    game.set_B_t(t, B_1)
    manual_B_1 = calculate_B1(game_Z_client_1)
    assert torch.allclose(B_1, manual_B_1, rtol=0.0, atol=1e-4)

    C_1 = game.calculate_c(t=1)
    game.set_C_t(t, C_1)
    manual_C1 = calculate_C1(game_Z_client_1)
    assert torch.allclose(C_1, manual_C1, rtol=0.0, atol=1e-3)

    D_1 = game.calculate_d(t=1)
    game.set_D_t(t, D_1)
    manual_D1 = calculate_D1(game_Z_client_1)
    assert torch.allclose(D_1, manual_D1, rtol=0.0, atol=1e-4)

    # The below function is checked manually
    e_alpha_gamma_A_inv = game.get_e_alpha_gamma_A_inv(1)

    initial_term = torch.matmul(bold_w_t_1, bold_w_t_1.T)
    wtyt = torch.matmul(bold_w_t_1, y_1)

    # Calculate P(1) manually : because alpha and gamma of clients are the same, they will have the same P
    manual_p1 = calculate_p1_manually(exp_var.alpha, exp_var.gamma, initial_term)
    manual_S1 = calculate_st_manually(exp_var.alpha, exp_var.gamma)
    for client_id in range(0, num_clients):
        client_pt = game.calculate_pt_client(
            1,
            client_id,
            e_alpha_gamma_A_inv,
            initial_term,
        )
        game.set_client_pt(1, client_id, pt_value=client_pt)
        assert torch.allclose(manual_p1, client_pt, rtol=0.0, atol=1e-4)

        client_st = game.calculate_st_client(
            1,
            client_id,
            e_alpha_gamma_A_inv,
            wtyt,
        )
        game.set_client_st(1, client_id, st_value=client_st)
        #  we only check 2 decimal places because so many manual errors are accumulated.
        assert torch.allclose(manual_S1, client_st, rtol=0.0, atol=1e-2)

    #  Compute game beta_1 (for all clients) --> shape: Nd_z x 1
    game_beta_1 = game.compute_beta(1, predictions_1)

    # Now compare previous beta with new beta
    for client_id in range(0, num_clients):
        old_beta = client_manager.clients[client_id].state.get_beta_t(1)
        new_beta = game_beta_1[client_id]
        assert old_beta.shape == new_beta.shape

    #  Our assumption
    game_beta_2 = game_beta_1
    new_predictions_3 = client_manager.get_predictions_with_beta(2, game_beta_2)

    # compute mixture weights again based on new predictions
    # Optional: the next two lines are optional. Update last predictions or not.
    #  --> if we update it, we can compute new server mixture.
    # update_prev_y = True
    # if update_prev_y:
    # new_predictions_2 = client_manager.get_predictions_with_beta(1, game_beta_1)
    # new_w_3 = server.compute_mixture_weights(new_predictions_2, y_2)
    # new_server_output_3 = torch.matmul(new_predictions_3, new_w_3)
    # We only need to again calculate w_3 if Y_2 has changed

    assert torch.allclose(past_T_betas_from_game[-1], game_beta_1, rtol=0.0, atol=1e-5)

    for client_id in range(0, num_clients):
        no_game_regret = compute_game_regret_objective(
            [game_beta_1[-1][client_id]],
            [target[3].reshape(exp_var.y_dim, 1)],
            [torch.matmul(predictions_3, w_3)],
            exp_var.gamma,
            exp_var.alpha,
            1,
        )
        game_no_Y_regret = compute_game_regret_objective(
            [game_beta_1[-1][client_id]],
            [target[3].reshape(exp_var.y_dim, 1)],
            [torch.matmul(new_predictions_3, w_3)],
            exp_var.gamma,
            exp_var.alpha,
            1,
        )
        # I think the below functionality does not make a lot of sense now.
        # game_regret = compute_game_regret_objective(
        #     [game_beta_1[-1][client_id]],
        #     [target[3].reshape(exp_var.y_dim, 1)],
        #     [torch.matmul(new_predictions_3, new_w_3)],
        #     exp_var.gamma,
        #     exp_var.alpha,
        #     0,
        # )

        assert (
            game_no_Y_regret < no_game_regret
        ), f"Failed for client {client_id} no_game_regret: {no_game_regret} < game_no_Y_regret: {game_no_Y_regret}"


def do_not_test_input_z_indices_in_game() -> None:
    seed = 2024
    random.seed(seed)
    torch.manual_seed(seed)
    T = 3
    z_dim = 2
    num_clients = 2
    data, target = set_data_target_long()
    exp_var = experiment_setup(1, z_dim, T, 0.1, 0.1, 0.001)

    client_manager = ClientManager(
        ClientType.RFN, num_clients, data, T, exp_var.z_dim, exp_var.alpha, exp_var.gamma, exp_var.sigma, target
    )
    game = RfnGame(
        client_manager.clients,
        sync_freq=exp_var.sync_freq,
        z_dim=exp_var.z_dim,
    )

    for t in range(0, 9):
        client_manager.clients[0].state.next_time_step(next_time=t)
        # First set fake hidden states : d_y x d_z.
        client_manager.clients[0].state.set_hidden_state(torch.randn((1, 2)), time=(t))

        if t % T == 0:
            game.init_game_round_variables(current_time=t)

            # manually map time between T to 0
            back_index = 0
            for back_t in range(T, -1, -1):
                index = int((t / T) * T - back_index)
                assert torch.allclose(
                    data[index], game.get_input(back_t, client_manager.clients[0]), rtol=0.0, atol=1e-5
                )
                back_index += 1

                client_hidden_state = client_manager.clients[0].state.get_hidden_state_t(index - 1)
                game_z = game.get_z(back_t - 1, client=client_manager.clients[0])

                assert torch.allclose(client_hidden_state, game_z, rtol=0.0, atol=1e-5)
