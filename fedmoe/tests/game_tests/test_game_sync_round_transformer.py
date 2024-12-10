import math

import torch

from fedmoe.game.transformer_game import TransformerGame
from fedmoe.server import Server
from fedmoe.tests.utils import get_data_and_target_sequences, get_transformer_client_manager

torch.set_default_dtype(torch.float64)
DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()
Z_DIM = 2
Y_DIM = 3  # This is fixed for this data
# T should be fixed in this test
T = 3
NUM_CLIENTS = 2
GAMMA = 5.0
check_game_regret = 1
check_game_residual = 1


def test_game_round_server() -> None:
    """
    This test checks that all the game matrices are calculated correctly starting from t = 0 to sync step.
    Also, it checks that residuals and the regret functions are reduced in all the steps after the game is played once.
    So, it passes if the game betas and game predictions in the synchronization step
      as well as previous steps are helpful.
    """
    torch.manual_seed(100)
    torch.set_default_dtype(torch.float64)

    client_manager = get_transformer_client_manager(Z_DIM, gamma=GAMMA, init_zero=True)

    client_manager.clients[0].gamma = GAMMA
    client_manager.clients[1].gamma = GAMMA

    game = TransformerGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=Z_DIM,
    )

    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=1.0,
        eta=1.0,
    )

    t = -1
    hat_Y_0 = client_manager.get_Y_0()

    w_neg1 = torch.randn((NUM_CLIENTS, 1)).double()
    w_neg1 = 1.0 * w_neg1 / torch.sum(w_neg1)
    _ = torch.matmul(w_neg1.T, hat_Y_0.double()).T

    # t = 0
    # next_predictions = client_manager.fit_clients(t)
    # w_t = server.compute_mixture_weights(hat_Y_0, TARGET_SEQUENCE[t].reshape(-1, 1))
    # assert w_t.shape == (NUM_CLIENTS, 1)
    # Play normally from t = 0 to t = T (3)
    mixture_weights = []
    client_predictions = [hat_Y_0]
    # no_game_residual_t stores the residual inner product for t from 0 to T-1 that is 0 , 1, and 2
    no_game_residuals = []

    for t in range(3):
        torch.set_default_dtype(torch.float64)
        next_predictions = client_manager.fit_clients(t)
        client_predictions.append(next_predictions)
        w_t = server.compute_mixture_weights(
            client_predictions[t].double(), TARGET_SEQUENCE[t].reshape(-1, 1).double()
        )
        assert w_t.shape == (NUM_CLIENTS, 1)
        mixture_weights.append(w_t)
        server_out_t = torch.matmul(w_t.T, next_predictions)
        no_game_residual_t = torch.pow(torch.linalg.norm(TARGET_SEQUENCE[t + 1].unsqueeze(1) - server_out_t.T), 2.0)
        no_game_residuals.append(no_game_residual_t)

    t = 3
    prediction_4 = client_manager.fit_clients(t)
    w_3 = server.compute_mixture_weights(client_predictions[3].double(), TARGET_SEQUENCE[t].reshape(-1, 1).double())
    assert w_3.shape == (NUM_CLIENTS, 1)
    mixture_weights.append(w_3)
    sync_step_residual = TARGET_SEQUENCE[4].unsqueeze(1) - torch.matmul(mixture_weights[3].T, prediction_4).T
    residual_inner_product_no = torch.pow(torch.linalg.norm(sync_step_residual), 2.0)
    T_regularizer_c0 = GAMMA * torch.pow(torch.linalg.norm(client_manager.clients[0].state.get_beta_t(3)), 2.0)
    T_regularizer_c1 = GAMMA * torch.pow(torch.linalg.norm(client_manager.clients[1].state.get_beta_t(3)), 2.0)

    regret_no_game = 2 * residual_inner_product_no + T_regularizer_c0 + T_regularizer_c1

    # Game needs observed values from zero to T inclusive
    torch.set_default_dtype(torch.float64)
    past_T_betas, past_game_predictions = server.sync_round(
        t,
        [
            TARGET_SEQUENCE[0].reshape(-1, 1).double(),
            TARGET_SEQUENCE[1].reshape(-1, 1).double(),
            TARGET_SEQUENCE[2].reshape(-1, 1).double(),
            TARGET_SEQUENCE[3].reshape(-1, 1).double(),
        ],
        [
            mixture_weights[0].double(),
            mixture_weights[1].double(),
            mixture_weights[2].double(),
            mixture_weights[3].double(),
        ],
    )
    # Improve step Ts predictions with the new beta_T <-- beta_(T-1)
    # past_T_betas has betas for t = 0, 1, 2
    # we say beta_3 = beta_2, and use beta_3 to make predictions for t = 4.
    # new_predictions_4 = client_manager.get_predictions_with_beta(t, past_T_betas[-1])

    # # before_game = torch.matmul(prediction_4, w_t)
    # # target = TARGET_SEQUENCE[4]
    # # after_game = torch.matmul(new_predictions_4, w_t)
    # # print(before_game, target, after_game)

    # # T = 3
    # # The below definition of regret is a bit different from the one in the paper (it is just the last step T).
    # last_betas = past_T_betas[-1].reshape(NUM_CLIENTS, Z_DIM, 1)
    # sync_step_residual = TARGET_SEQUENCE[4] - torch.matmul(new_predictions_4, mixture_weights[3])
    # residual_inner_product_game = torch.pow(torch.linalg.norm(sync_step_residual), 2.0)

    # T_regularizer_c0 = GAMMA * torch.pow(torch.linalg.norm(last_betas[0]), 2.0)
    # T_regularizer_c1 = GAMMA * torch.pow(torch.linalg.norm(last_betas[1]), 2.0)

    # regret_game = 2*residual_inner_product_game + T_regularizer_c0 + T_regularizer_c1

    # print("main residual no game", residual_inner_product_no)
    # print("main residual game", residual_inner_product_game)

    # print("game betas", last_betas)
    # T-1 = 2
    clients = game.clients

    # Checking A hat for T-1
    mixture_weights_2 = mixture_weights[2]
    bold_w_2 = game.create_bold_w_t(mixture_weights_2)
    W_2_W_2_T = torch.matmul(bold_w_2, bold_w_2.T).double()
    manual_A_hat_2 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for i in range(NUM_CLIENTS):
        # here we don't need to convert times because it is the first game round
        Z_t = clients[i].state.get_hidden_state_t(2).double()
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        for j in range(NUM_CLIENTS):
            Z_t_j = clients[j].state.get_hidden_state_t(2).double()
            e_j = clients[j].get_e(NUM_CLIENTS).double()
            manual_A_hat_2[i, j] = torch.matmul(
                torch.matmul(torch.matmul(Z_t.T, e_i.T), W_2_W_2_T), torch.matmul(e_j, Z_t_j)
            )
    manual_A_hat_2 = manual_A_hat_2.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(game.game_state.get_A_hat_t(2), manual_A_hat_2, rtol=0.0, atol=1e-5)

    # Checking D for T-1
    manual_D_2_list = []
    for i in range(NUM_CLIENTS):
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        Z_i = clients[i].state.get_hidden_state_t(2).double()
        manual_D_2_list.append(torch.matmul(e_i, Z_i))

    manual_D_2 = torch.cat(manual_D_2_list, dim=1)
    assert torch.allclose(game.game_state.get_D_t(2), manual_D_2, rtol=0.0, atol=1e-5)

    # Checking G hat for T-1 (needs D_2 and \hat{A}_2)
    Iz = torch.eye(Z_DIM).double()

    alpha_tensor = torch.exp(torch.Tensor([-1 * client.alpha * 0.0 for client in clients]))
    e_t_2 = torch.block_diag(*[alpha * Iz for alpha in alpha_tensor])
    bold_gamma = torch.block_diag(*[client.gamma * Iz for client in clients])
    manual_G_2_part1 = torch.linalg.inv(
        -1 * torch.add(torch.matmul(e_t_2, bold_gamma), torch.matmul(e_t_2, game.game_state.get_A_hat_t(2)))
    )
    manual_G_2 = torch.matmul(
        manual_G_2_part1, torch.matmul(torch.matmul(e_t_2, game.game_state.get_D_t(2).T), W_2_W_2_T)
    )
    assert torch.allclose(game.game_state.get_G_t(2), manual_G_2, rtol=0.0, atol=1e-5)

    # Checking H for T-1
    # print("game state last H", game.game_state.get_H_t(2))
    e_t_2_gamma = torch.matmul(e_t_2, bold_gamma)
    manual_H_2_part1 = torch.add(
        torch.add(e_t_2_gamma, game.game_state.get_A_t(2)),
        torch.matmul(e_t_2, game.game_state.get_A_hat_t(2)),
    )

    manual_H_part_2 = torch.matmul(
        torch.matmul(torch.matmul(e_t_2, game.game_state.get_D_t(2).double().T), bold_w_2.double()),
        TARGET_SEQUENCE[3].reshape(-1, 1).double(),
    ) - game.game_state.get_C_t(2)

    # inv_matrix = torch.linalg.pinv(manual_H_2_part1.to(torch.float64))
    manual_H_2 = torch.matmul(torch.linalg.inv(manual_H_2_part1), manual_H_part_2)

    assert torch.allclose(game.game_state.get_H_t(2), manual_H_2, rtol=0.0, atol=1e-5)

    # # A, B, C, and D_i for T-1 should be zero.
    manual_A_2 = torch.zeros((NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)).double()
    assert torch.allclose(game.game_state.get_A_t(2), manual_A_2, rtol=0.0, atol=1e-5)
    manual_B_2 = torch.zeros((NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Y_DIM)).double()
    assert torch.allclose(game.game_state.get_B_t(2), manual_B_2, rtol=0.0, atol=1e-5)
    manual_C_2 = torch.zeros((NUM_CLIENTS * Z_DIM, 1)).double()
    assert torch.allclose(game.game_state.get_C_t(2), manual_C_2, rtol=0.0, atol=1e-5)

    # Checking D_i_2 for client 0 and client 1
    manual_D_c0_2 = torch.zeros((NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)).double()
    assert torch.allclose(clients[0].D[2], manual_D_c0_2, rtol=0.0, atol=1e-5)
    manual_D_c1_2 = torch.zeros((NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)).double()
    assert torch.allclose(clients[1].D[2], manual_D_c1_2, rtol=0.0, atol=1e-5)

    # Checking P and S for t=2 (T-1)
    # P(T) and S(T) are zero (t=3)
    # Client 0
    P_c0_3 = torch.zeros((NUM_CLIENTS * Y_DIM, NUM_CLIENTS * Y_DIM)).double()
    S_c0_3 = torch.zeros((NUM_CLIENTS * Y_DIM, 1)).double()
    assert torch.allclose(clients[0].P[3], P_c0_3, rtol=0.0, atol=1e-5)
    assert torch.allclose(clients[0].S[3], S_c0_3, rtol=0.0, atol=1e-5)

    # Manually compute P and S for t=2 (T-1) for client 0
    e_hat_c0 = clients[0].get_hat_e(NUM_CLIENTS).double()
    e_hat_c0_e_hat_c0 = torch.matmul(e_hat_c0, e_hat_c0.T).double()
    P_c0_2_part1 = torch.matmul(
        torch.matmul(
            manual_G_2.T.double(), torch.add(manual_A_hat_2.double(), clients[0].gamma * e_hat_c0_e_hat_c0.double())
        ),
        manual_G_2.double(),
    ).double()

    line2_manual = torch.matmul(torch.matmul(W_2_W_2_T, manual_D_2.double()), manual_G_2.double())
    line_3_manual = torch.matmul(torch.matmul(manual_G_2.T.double(), manual_D_2.T.double()), W_2_W_2_T.T.double())
    manual_P_c0_2 = P_c0_2_part1 + line2_manual + line_3_manual + W_2_W_2_T.double()
    manual_P_c0_2 = manual_P_c0_2.reshape(NUM_CLIENTS * Y_DIM, NUM_CLIENTS * Y_DIM)

    assert torch.allclose(clients[0].P[2], manual_P_c0_2, rtol=0.0, atol=1e-4)

    # S for client 0
    S_c0_2_line1 = torch.matmul(
        torch.matmul(manual_G_2.T, (manual_A_hat_2 + clients[0].gamma * e_hat_c0_e_hat_c0.double())), manual_H_2
    )
    S_c0_2_line2 = torch.matmul(torch.matmul(W_2_W_2_T, manual_D_2), manual_H_2)
    S_c0_2_line3 = torch.matmul(
        torch.matmul(manual_G_2.T, manual_D_2.T).double(),
        -1 * torch.matmul(bold_w_2.double(), TARGET_SEQUENCE[3].reshape(-1, 1).double()),
    )
    S_c0_2_line4 = -1 * torch.matmul(bold_w_2, TARGET_SEQUENCE[3].reshape(-1, 1).double())

    manual_S_c0_2 = S_c0_2_line1 + S_c0_2_line2 + S_c0_2_line3 + S_c0_2_line4

    assert torch.allclose(clients[0].S[2].detach(), manual_S_c0_2.detach(), rtol=0.0, atol=1e-5)

    # Client 1
    P_c1_3 = torch.zeros((NUM_CLIENTS * Y_DIM, NUM_CLIENTS * Y_DIM)).double()
    S_c1_3 = torch.zeros((NUM_CLIENTS * Y_DIM, 1)).double()
    assert torch.allclose(clients[1].P[3], P_c1_3, rtol=0.0, atol=1e-5)
    assert torch.allclose(clients[1].S[3], S_c1_3, rtol=0.0, atol=1e-5)

    # Manually compute P and S for t=2 (T-1) for client 0
    e_hat_c1 = clients[1].get_hat_e(NUM_CLIENTS).double()
    e_hat_c1_e_hat_c1 = torch.matmul(e_hat_c1, e_hat_c1.T).double()
    P_c1_2_part1 = torch.matmul(
        torch.matmul(
            manual_G_2.T.double(), torch.add(manual_A_hat_2.double(), clients[1].gamma * e_hat_c1_e_hat_c1.double())
        ),
        manual_G_2.double(),
    ).double()

    line2_manual = torch.matmul(torch.matmul(W_2_W_2_T, manual_D_2.double()), manual_G_2.double())
    line_3_manual = torch.matmul(torch.matmul(manual_G_2.T.double(), manual_D_2.T.double()), W_2_W_2_T.T.double())
    manual_P_c1_2 = P_c1_2_part1 + line2_manual + line_3_manual + W_2_W_2_T.double()
    manual_P_c1_2 = manual_P_c1_2.reshape(NUM_CLIENTS * Y_DIM, NUM_CLIENTS * Y_DIM)

    assert torch.allclose(clients[1].P[2], manual_P_c1_2, rtol=0.0, atol=1e-4)

    # S for client 1
    S_c1_2_line1 = torch.matmul(
        torch.matmul(manual_G_2.T, (manual_A_hat_2 + clients[1].gamma * e_hat_c1_e_hat_c1.double())), manual_H_2
    )
    S_c1_2_line2 = torch.matmul(torch.matmul(W_2_W_2_T, manual_D_2), manual_H_2)
    S_c1_2_line3 = torch.matmul(
        torch.matmul(manual_G_2.T, manual_D_2.T).double(),
        -1 * torch.matmul(bold_w_2.double(), TARGET_SEQUENCE[3].reshape(-1, 1).double()),
    )
    S_c1_2_line4 = -1 * torch.matmul(bold_w_2, TARGET_SEQUENCE[3].reshape(-1, 1).double())

    manual_S_c1_2 = S_c1_2_line1 + S_c1_2_line2 + S_c1_2_line3 + S_c1_2_line4
    assert torch.allclose(clients[1].S[2], manual_S_c1_2, rtol=0.0, atol=1e-5)

    # <----------> Decreasing t from T-1 to T-2 (2 to 1) in the game loop<----------> ######
    # Checking A hat for T-2 (t=1)
    mixture_weights_1 = mixture_weights[1]
    bold_w_1 = game.create_bold_w_t(mixture_weights_1)
    W_1_W_1_T = torch.matmul(bold_w_1, bold_w_1.T).double()
    manual_A_hat_1 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for i in range(NUM_CLIENTS):
        # here we don't need to convert times because it is the first game round
        Z_1 = clients[i].state.get_hidden_state_t(1).double()
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        for j in range(NUM_CLIENTS):
            Z_1_j = clients[j].state.get_hidden_state_t(1).double()
            e_j = clients[j].get_e(NUM_CLIENTS).double()
            manual_A_hat_1[i, j] = torch.matmul(
                torch.matmul(torch.matmul(Z_1.T, e_i.T), W_1_W_1_T), torch.matmul(e_j, Z_1_j)
            )
    manual_A_hat_1 = manual_A_hat_1.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(game.game_state.get_A_hat_t(1), manual_A_hat_1, rtol=0.0, atol=1e-5)

    # Checking D for T-2 (t=1)
    manual_D_1_list = []
    for i in range(NUM_CLIENTS):
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        Z_i = clients[i].state.get_hidden_state_t(1).double()
        manual_D_1_list.append(torch.matmul(e_i, Z_i))

    manual_D_1 = torch.cat(manual_D_1_list, dim=1)
    assert torch.allclose(game.game_state.get_D_t(1), manual_D_1, rtol=0.0, atol=1e-5)

    # Checking A_1
    manual_A_1 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for i, client_i in enumerate(clients):
        for j, client_j in enumerate(clients):
            manual_A_1[i, j] = torch.matmul(
                torch.matmul(
                    torch.matmul(
                        client_i.state.get_hidden_state_t(1).T.double(), client_i.get_e(NUM_CLIENTS).T.double()
                    ),
                    client_i.P[2].double(),
                ),
                torch.matmul(client_j.get_e(NUM_CLIENTS).double(), client_j.state.get_hidden_state_t(1).double()),
            )
    manual_A_1 = manual_A_1.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(game.game_state.get_A_t(1), manual_A_1, rtol=0.0, atol=1e-5)

    # Checking B_1
    manual_B_1_list = []
    for client_i in clients:
        manual_B_1_list.append(
            torch.matmul(
                torch.matmul(client_i.P[2], client_i.get_e(NUM_CLIENTS)), client_i.state.get_hidden_state_t(1)
            )
        )
    manual_B_1 = torch.cat(manual_B_1_list, dim=1).T

    assert torch.allclose(game.game_state.get_B_t(1), manual_B_1, rtol=0.0, atol=1e-5)

    # Checking C_1
    manual_C_1_list = []
    for client_i in clients:
        manual_C_1_list.append(
            torch.matmul(
                torch.matmul(client_i.S[2].T, client_i.get_e(NUM_CLIENTS)), client_i.state.get_hidden_state_t(1)
            )
        )
    manual_C_1 = torch.cat(manual_C_1_list, dim=1).T
    assert torch.allclose(game.game_state.get_C_t(1), manual_C_1, rtol=0.0, atol=1e-5)

    # Checking G hat for T-2 (t=1) (needs D_1 and \hat{A}_1, A_1, and B_1)

    Iz = torch.eye(Z_DIM).double()
    alpha_tensor = torch.exp(torch.Tensor([-1 * client.alpha for client in clients]))
    exp_neg_alpha = torch.block_diag(*[alpha * Iz for alpha in alpha_tensor])
    bold_gamma = torch.block_diag(*[client.gamma * Iz for client in clients])

    manual_G_1_part1 = -1 * torch.inverse(
        torch.matmul(exp_neg_alpha, bold_gamma) + manual_A_1 + torch.matmul(exp_neg_alpha, manual_A_hat_1)
    )
    manual_G_1_part2 = torch.matmul(torch.matmul(exp_neg_alpha, game.game_state.get_D_t(1).T), W_1_W_1_T) + manual_B_1
    manual_G_1 = torch.matmul(manual_G_1_part1, manual_G_1_part2)

    assert torch.allclose(game.game_state.get_G_t(1), manual_G_1, rtol=0.0, atol=1e-5)

    # Checking H for T-2 (t=1)
    e_t_1_gamma = torch.matmul(exp_neg_alpha, bold_gamma)
    manual_H_1_part1 = torch.add(torch.add(e_t_1_gamma, manual_A_1), torch.matmul(exp_neg_alpha, manual_A_hat_1))
    manual_H_1_part2 = (
        torch.matmul(
            torch.matmul(torch.matmul(exp_neg_alpha, game.game_state.get_D_t(1).double().T), bold_w_1.double()),
            TARGET_SEQUENCE[2].reshape(-1, 1).double(),
        )
        - manual_C_1
    )
    manual_H_1 = torch.matmul(torch.inverse(manual_H_1_part1), manual_H_1_part2)

    assert torch.allclose(game.game_state.get_H_t(1), manual_H_1, rtol=0.0, atol=1e-5)

    # Checking D_i_1 for client 0 and client 1
    manual_D_c0_1 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()

    for client in clients:
        e_i = client.get_e(NUM_CLIENTS).double()
        Z_i = client.state.get_hidden_state_t(1).double()
        first_term = torch.matmul(torch.matmul(e_i, Z_i).T, clients[0].P[2].double())
        for client_j in clients:
            e_j = client_j.get_e(NUM_CLIENTS).double()
            Z_j = client_j.state.get_hidden_state_t(1).double()
            item = torch.matmul(first_term, torch.matmul(e_j, Z_j))
            manual_D_c0_1[client.id][client_j.id] = item

    manual_D_c0_1 = manual_D_c0_1.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)

    assert torch.allclose(clients[0].D[1], manual_D_c0_1, rtol=0.0, atol=1e-5)

    # D_1 for client 1
    manual_D_c1_1 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for client in clients:
        e_i = client.get_e(NUM_CLIENTS).double()
        Z_i = client.state.get_hidden_state_t(1).double()
        first_term = torch.matmul(torch.matmul(e_i, Z_i).T, clients[1].P[2].double())
        for client_j in clients:
            e_j = client_j.get_e(NUM_CLIENTS).double()
            Z_j = client_j.state.get_hidden_state_t(1).double()
            item = torch.matmul(first_term, torch.matmul(e_j, Z_j))
            manual_D_c1_1[client.id][client_j.id] = item

    manual_D_c1_1 = manual_D_c1_1.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(clients[1].D[1], manual_D_c1_1, rtol=0.0, atol=1e-5)

    # Checking P and S for t=1 (T-2)
    # Client 0
    exp_neg_alpha_c0 = math.exp(-1 * clients[0].alpha)
    P_c0_1_line1 = torch.matmul(
        torch.matmul(
            manual_G_1.T,
            torch.add(manual_A_hat_1, manual_D_c0_1) + exp_neg_alpha_c0 * clients[0].gamma * e_hat_c0_e_hat_c0,
        ),
        manual_G_1,
    )

    P_c0_1_line2 = torch.matmul(torch.matmul((exp_neg_alpha_c0 * W_1_W_1_T + clients[0].P[2]), manual_D_1), manual_G_1)
    P_c0_1_line3 = torch.matmul(
        torch.matmul(manual_G_1.T, manual_D_1.T), (exp_neg_alpha_c0 * W_1_W_1_T.T + clients[0].P[2]).T
    )
    P_c0_1_line4 = exp_neg_alpha_c0 * W_1_W_1_T + clients[0].P[2]

    manual_P_c0_1 = P_c0_1_line1 + P_c0_1_line2 + P_c0_1_line3 + P_c0_1_line4
    assert torch.allclose(clients[0].P[1], manual_P_c0_1, rtol=0.0, atol=1e-4)

    # P_1 for client 1
    exp_neg_alpha_c1 = math.exp(-1 * clients[1].alpha)
    P_c1_1_line1 = torch.matmul(
        torch.matmul(
            manual_G_1.T,
            torch.add(manual_A_hat_1, manual_D_c1_1) + exp_neg_alpha_c1 * clients[1].gamma * e_hat_c1_e_hat_c1,
        ),
        manual_G_1,
    )

    P_c1_1_line2 = torch.matmul(torch.matmul((exp_neg_alpha_c1 * W_1_W_1_T + clients[1].P[2]), manual_D_1), manual_G_1)
    P_c1_1_line3 = torch.matmul(
        torch.matmul(manual_G_1.T, manual_D_1.T), (exp_neg_alpha_c1 * W_1_W_1_T.T + clients[1].P[2]).T
    )
    P_c1_1_line4 = exp_neg_alpha_c1 * W_1_W_1_T + clients[1].P[2]

    manual_P_c1_1 = P_c1_1_line1 + P_c1_1_line2 + P_c1_1_line3 + P_c1_1_line4
    assert torch.allclose(clients[1].P[1], manual_P_c1_1, rtol=0.0, atol=1e-4)

    # Check S_1 for client 0
    manual_S_c0_1_line1 = (
        manual_G_1.T
        @ (manual_A_hat_1 + manual_D_c0_1 + exp_neg_alpha_c0 * clients[0].gamma * e_hat_c0_e_hat_c0)
        @ manual_H_1
    )
    manual_S_c0_1_line2 = (exp_neg_alpha_c0 * W_1_W_1_T + clients[0].P[2]) @ manual_D_1 @ manual_H_1
    manual_S_c0_1_line3 = (
        manual_G_1.T
        @ manual_D_1.T
        @ (-1 * bold_w_1 @ TARGET_SEQUENCE[2].reshape(-1, 1) * exp_neg_alpha_c0 + clients[0].S[2])
    )
    manual_S_c0_1_line4 = -1 * bold_w_1 @ TARGET_SEQUENCE[2].reshape(-1, 1) * exp_neg_alpha_c0 + clients[0].S[2]

    manual_S_c0_1 = manual_S_c0_1_line1 + manual_S_c0_1_line2 + manual_S_c0_1_line3 + manual_S_c0_1_line4
    assert torch.allclose(clients[0].S[1], manual_S_c0_1, rtol=0.0, atol=1e-5)

    # Check S_1 for client 1
    manual_S_c1_1_line1 = (
        manual_G_1.T
        @ (manual_A_hat_1 + manual_D_c1_1 + exp_neg_alpha_c1 * clients[1].gamma * e_hat_c1_e_hat_c1)
        @ manual_H_1
    )
    manual_S_c1_1_line2 = (exp_neg_alpha_c1 * W_1_W_1_T + clients[1].P[2]) @ manual_D_1 @ manual_H_1
    manual_S_c1_1_line3 = (
        manual_G_1.T
        @ manual_D_1.T
        @ (-1 * bold_w_1 @ TARGET_SEQUENCE[2].reshape(-1, 1) * exp_neg_alpha_c1 + clients[1].S[2])
    )
    manual_S_c1_1_line4 = -1 * bold_w_1 @ TARGET_SEQUENCE[2].reshape(-1, 1) * exp_neg_alpha_c1 + clients[1].S[2]

    manual_S_c1_1 = manual_S_c1_1_line1 + manual_S_c1_1_line2 + manual_S_c1_1_line3 + manual_S_c1_1_line4
    assert torch.allclose(clients[1].S[1], manual_S_c1_1, rtol=0.0, atol=1e-5)

    # Now move to T-3 in the inner loop in the game.
    # Checking A hat for T-3 (t=0)
    mixture_weights_0 = mixture_weights[0]
    bold_w_0 = game.create_bold_w_t(mixture_weights_0)
    W_0_W_0_T = torch.matmul(bold_w_0, bold_w_0.T).double()
    manual_A_hat_0 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for i in range(NUM_CLIENTS):
        # here we don't need to convert times because it is the first game round
        Z_0 = clients[i].state.get_hidden_state_t(0).double()
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        for j in range(NUM_CLIENTS):
            Z_0_j = clients[j].state.get_hidden_state_t(0).double()
            e_j = clients[j].get_e(NUM_CLIENTS).double()
            manual_A_hat_0[i, j] = torch.matmul(
                torch.matmul(torch.matmul(Z_0.T, e_i.T), W_0_W_0_T), torch.matmul(e_j, Z_0_j)
            )
    manual_A_hat_0 = manual_A_hat_0.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(game.game_state.get_A_hat_t(0), manual_A_hat_0, rtol=0.0, atol=1e-5)

    # Checking D for T-3 (t=0)
    manual_D_0_list = []
    for i in range(NUM_CLIENTS):
        e_i = clients[i].get_e(NUM_CLIENTS).double()
        Z_i = clients[i].state.get_hidden_state_t(0).double()
        manual_D_0_list.append(torch.matmul(e_i, Z_i))

    manual_D_0 = torch.cat(manual_D_0_list, dim=1)
    assert torch.allclose(game.game_state.get_D_t(0), manual_D_0, rtol=0.0, atol=1e-5)

    # Checking A_0
    manual_A_0 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for i, client_i in enumerate(clients):
        for j, client_j in enumerate(clients):
            manual_A_0[i, j] = torch.matmul(
                torch.matmul(
                    torch.matmul(
                        client_i.state.get_hidden_state_t(0).T.double(), client_i.get_e(NUM_CLIENTS).T.double()
                    ),
                    client_i.P[1].double(),
                ),
                torch.matmul(client_j.get_e(NUM_CLIENTS).double(), client_j.state.get_hidden_state_t(0).double()),
            )
    manual_A_0 = manual_A_0.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(game.game_state.get_A_t(0), manual_A_0, rtol=0.0, atol=1e-5)

    # Checking B_0
    manual_B_0_list = []
    for client_i in clients:
        manual_B_0_list.append(
            torch.matmul(
                torch.matmul(client_i.P[1], client_i.get_e(NUM_CLIENTS)), client_i.state.get_hidden_state_t(0)
            )
        )
    manual_B_0 = torch.cat(manual_B_0_list, dim=1).T

    assert torch.allclose(game.game_state.get_B_t(0), manual_B_0, rtol=0.0, atol=1e-5)

    # Checking C_0
    manual_C_0_list = []
    for client_i in clients:
        manual_C_0_list.append(
            torch.matmul(
                torch.matmul(client_i.S[1].T, client_i.get_e(NUM_CLIENTS)), client_i.state.get_hidden_state_t(0)
            )
        )
    manual_C_0 = torch.cat(manual_C_0_list, dim=1).T
    assert torch.allclose(game.game_state.get_C_t(0), manual_C_0, rtol=0.0, atol=1e-5)

    # Checking G hat for T-3 (t=0) (needs D_0 and \hat{A}_0, A_0, and B_0)
    Iz = torch.eye(Z_DIM).double()
    alpha_tensor = torch.exp(torch.Tensor([-1 * 2 * client.alpha for client in clients]))
    exp_neg_2alpha = torch.block_diag(*[alpha * Iz for alpha in alpha_tensor])
    bold_gamma = torch.block_diag(*[client.gamma * Iz for client in clients])

    manual_G_0_part1 = -1 * torch.inverse(
        torch.matmul(exp_neg_2alpha, bold_gamma) + manual_A_0 + torch.matmul(exp_neg_2alpha, manual_A_hat_0)
    )
    manual_G_0_part2 = torch.matmul(torch.matmul(exp_neg_2alpha, game.game_state.get_D_t(0).T), W_0_W_0_T) + manual_B_0
    manual_G_0 = torch.matmul(manual_G_0_part1, manual_G_0_part2)

    assert torch.allclose(game.game_state.get_G_t(0), manual_G_0, rtol=0.0, atol=1e-5)

    # Checking H for T-3 (t=0)
    e_t_0_gamma = torch.matmul(exp_neg_2alpha, bold_gamma)
    manual_H_0_part1 = torch.add(torch.add(e_t_0_gamma, manual_A_0), torch.matmul(exp_neg_2alpha, manual_A_hat_0))
    manual_H_0_part2 = (
        torch.matmul(
            torch.matmul(torch.matmul(exp_neg_2alpha, game.game_state.get_D_t(0).double().T), bold_w_0.double()),
            TARGET_SEQUENCE[1].reshape(-1, 1).double(),
        )
        - manual_C_0
    )
    manual_H_0 = torch.matmul(torch.inverse(manual_H_0_part1), manual_H_0_part2)

    assert torch.allclose(game.game_state.get_H_t(0), manual_H_0, rtol=0.0, atol=1e-5)

    # Checking D_i_0 for client 0 and client 1
    manual_D_c0_0 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()

    for client in clients:
        e_i = client.get_e(NUM_CLIENTS).double()
        Z_i = client.state.get_hidden_state_t(0).double()
        first_term = torch.matmul(torch.matmul(e_i, Z_i).T, clients[0].P[1].double())
        for client_j in clients:
            e_j = client_j.get_e(NUM_CLIENTS).double()
            Z_j = client_j.state.get_hidden_state_t(0).double()
            item = torch.matmul(first_term, torch.matmul(e_j, Z_j))
            manual_D_c0_0[client.id][client_j.id] = item

    manual_D_c0_0 = manual_D_c0_0.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)

    assert torch.allclose(clients[0].D[0], manual_D_c0_0, rtol=0.0, atol=1e-5)

    # D_0 for client 1
    manual_D_c1_0 = torch.zeros((NUM_CLIENTS, NUM_CLIENTS, Z_DIM, Z_DIM)).double()
    for client in clients:
        e_i = client.get_e(NUM_CLIENTS).double()
        Z_i = client.state.get_hidden_state_t(0).double()
        first_term = torch.matmul(torch.matmul(e_i, Z_i).T, clients[1].P[1].double())
        for client_j in clients:
            e_j = client_j.get_e(NUM_CLIENTS).double()
            Z_j = client_j.state.get_hidden_state_t(0).double()
            item = torch.matmul(first_term, torch.matmul(e_j, Z_j))
            manual_D_c1_0[client.id][client_j.id] = item

    manual_D_c1_0 = manual_D_c1_0.reshape(NUM_CLIENTS * Z_DIM, NUM_CLIENTS * Z_DIM)
    assert torch.allclose(clients[1].D[0], manual_D_c1_0, rtol=0.0, atol=1e-5)

    # Checking P and S for t=0 (T-3)
    # Client 0
    exp_2neg_alpha_c0 = math.exp(-1 * 2 * clients[0].alpha)
    P_c0_0_line1 = torch.matmul(
        torch.matmul(
            manual_G_0.T,
            torch.add(manual_A_hat_0, manual_D_c0_0) + exp_2neg_alpha_c0 * clients[0].gamma * e_hat_c0_e_hat_c0,
        ),
        manual_G_0,
    )

    P_c0_0_line2 = torch.matmul(
        torch.matmul((exp_2neg_alpha_c0 * W_0_W_0_T + clients[0].P[1]), manual_D_0), manual_G_0
    )
    P_c0_0_line3 = torch.matmul(
        torch.matmul(manual_G_0.T, manual_D_0.T), (exp_2neg_alpha_c0 * W_0_W_0_T.T + clients[0].P[1]).T
    )
    P_c0_0_line4 = exp_2neg_alpha_c0 * W_0_W_0_T + clients[0].P[1]

    manual_P_c0_0 = P_c0_0_line1 + P_c0_0_line2 + P_c0_0_line3 + P_c0_0_line4
    assert torch.allclose(clients[0].P[0], manual_P_c0_0, rtol=0.0, atol=1e-5)

    # P_0 for client 1
    exp_2neg_alpha_c1 = math.exp(-1 * 2 * clients[1].alpha)
    P_c1_0_line1 = torch.matmul(
        torch.matmul(
            manual_G_0.T,
            torch.add(manual_A_hat_0, manual_D_c1_0) + exp_2neg_alpha_c1 * clients[1].gamma * e_hat_c1_e_hat_c1,
        ),
        manual_G_0,
    )

    P_c1_0_line2 = torch.matmul(
        torch.matmul((exp_2neg_alpha_c1 * W_0_W_0_T + clients[1].P[1]), manual_D_0), manual_G_0
    )
    P_c1_0_line3 = torch.matmul(
        torch.matmul(manual_G_0.T, manual_D_0.T), (exp_2neg_alpha_c1 * W_0_W_0_T.T + clients[1].P[1]).T
    )
    P_c1_0_line4 = exp_2neg_alpha_c1 * W_0_W_0_T + clients[1].P[1]

    manual_P_c1_0 = P_c1_0_line1 + P_c1_0_line2 + P_c1_0_line3 + P_c1_0_line4
    assert torch.allclose(clients[1].P[0], manual_P_c1_0, rtol=0.0, atol=1e-5)

    # Check S_0 for client 0
    manual_S_c0_0_line1 = (
        manual_G_0.T
        @ (manual_A_hat_0 + manual_D_c0_0 + exp_2neg_alpha_c0 * clients[0].gamma * e_hat_c0_e_hat_c0)
        @ manual_H_0
    )
    manual_S_c0_0_line2 = (exp_2neg_alpha_c0 * W_0_W_0_T + clients[0].P[1]) @ manual_D_0 @ manual_H_0
    manual_S_c0_0_line3 = (
        manual_G_0.T
        @ manual_D_0.T
        @ (-1 * bold_w_0 @ TARGET_SEQUENCE[1].reshape(-1, 1) * exp_2neg_alpha_c0 + clients[0].S[1])
    )
    manual_S_c0_0_line4 = -1 * bold_w_0 @ TARGET_SEQUENCE[1].reshape(-1, 1) * exp_2neg_alpha_c0 + clients[0].S[1]

    manual_S_c0_0 = manual_S_c0_0_line1 + manual_S_c0_0_line2 + manual_S_c0_0_line3 + manual_S_c0_0_line4
    assert torch.allclose(clients[0].S[0], manual_S_c0_0, rtol=0.0, atol=1e-5)

    # Check S_0 for client 1
    manual_S_c1_0_line1 = (
        manual_G_0.T
        @ (manual_A_hat_0 + manual_D_c1_0 + exp_2neg_alpha_c1 * clients[1].gamma * e_hat_c1_e_hat_c1)
        @ manual_H_0
    )
    manual_S_c1_0_line2 = (exp_2neg_alpha_c1 * W_0_W_0_T + clients[1].P[1]) @ manual_D_0 @ manual_H_0
    manual_S_c1_0_line3 = (
        manual_G_0.T
        @ manual_D_0.T
        @ (-1 * bold_w_0 @ TARGET_SEQUENCE[1].reshape(-1, 1) * exp_2neg_alpha_c1 + clients[1].S[1])
    )
    manual_S_c1_0_line4 = -1 * bold_w_0 @ TARGET_SEQUENCE[1].reshape(-1, 1) * exp_2neg_alpha_c1 + clients[1].S[1]

    manual_S_c1_0 = manual_S_c1_0_line1 + manual_S_c1_0_line2 + manual_S_c1_0_line3 + manual_S_c1_0_line4
    assert torch.allclose(clients[1].S[0], manual_S_c1_0, rtol=0.0, atol=1e-5)

    # ##### now that we have all the matrices for the first round of the game, we can compute optimal betas.

    Y_0 = torch.cat([torch.Tensor(TARGET_SEQUENCE[0].reshape(-1, 1).double()) for client in clients], dim=0)
    assert Y_0.shape == (NUM_CLIENTS * Y_DIM, 1)

    # Second for loop in game (forward loop) for 0 to T-1
    t = 0
    beta_game_0 = manual_G_0 @ Y_0 + manual_H_0
    hidden_state_vector = [clients[0].state.get_hidden_state_t(0), clients[1].state.get_hidden_state_t(0)]
    hidden_states_0 = torch.block_diag(*[state for state in hidden_state_vector])
    game_Y_1 = torch.add(Y_0, hidden_states_0 @ beta_game_0)
    assert game_Y_1.shape == (NUM_CLIENTS * Y_DIM, 1)

    t = 1
    beta_game_1 = manual_G_1 @ game_Y_1 + manual_H_1
    hidden_state_vector = [clients[0].state.get_hidden_state_t(1), clients[1].state.get_hidden_state_t(1)]
    hidden_states_1 = torch.block_diag(*[state for state in hidden_state_vector])
    game_Y_2 = torch.add(game_Y_1, hidden_states_1 @ beta_game_1)
    assert game_Y_2.shape == (NUM_CLIENTS * Y_DIM, 1)

    t = 2
    beta_game_2 = manual_G_2 @ game_Y_2 + manual_H_2
    hidden_state_vector = [clients[0].state.get_hidden_state_t(2), clients[1].state.get_hidden_state_t(2)]
    hidden_states_2 = torch.block_diag(*[state for state in hidden_state_vector])
    game_Y_3 = torch.add(game_Y_2, hidden_states_2 @ beta_game_2)
    assert game_Y_3.shape == (NUM_CLIENTS * Y_DIM, 1)

    # Make sure these betas are the same ones as computed in the server game
    assert torch.allclose(past_T_betas[0], beta_game_0, rtol=0.0, atol=1e-5)
    assert torch.allclose(past_T_betas[1], beta_game_1, rtol=0.0, atol=1e-5)
    assert torch.allclose(past_T_betas[2], beta_game_2, rtol=0.0, atol=1e-5)

    assert torch.allclose(Y_0, past_game_predictions[0], rtol=0.0, atol=1e-5)
    assert torch.allclose(game_Y_1, past_game_predictions[1], rtol=0.0, atol=1e-5)
    assert torch.allclose(game_Y_2, past_game_predictions[2], rtol=0.0, atol=1e-5)
    assert torch.allclose(game_Y_3, past_game_predictions[3], rtol=0.0, atol=1e-5)

    # Now we want to see if these new betas are better than the one computed previously on the server.
    beta_game_2 = beta_game_2.reshape(NUM_CLIENTS, Z_DIM, 1)
    # We are at step T = 3, and the result of the game is, beta_2 that we use instead of beta_3
    # This function updates the beta and new prediction in each client
    manual_game_prediction_4_c0 = past_game_predictions[-1].reshape(NUM_CLIENTS, Y_DIM, 1)[0].reshape(-1, 1) + (
        clients[0].state.get_hidden_state_t(3) @ beta_game_2[0]
    )
    # After debugging we found out we need to update previous prediction based on the game
    client_manager.improve_previous_predictions_from_game(3, past_game_predictions[-1], past_T_betas[-1])
    game_prediction_4_c0 = clients[0].update_prediction_with_beta(3, beta_game_2[0])
    # Check new predictions calculations
    assert torch.allclose(manual_game_prediction_4_c0, game_prediction_4_c0, rtol=0.0, atol=1e-5)

    # Repeat for client 1, now we have updated its previous Y
    manual_game_prediction_4_c1 = clients[1].state.get_prediction_t(3) + (
        clients[1].state.get_hidden_state_t(3) @ beta_game_2[1]
    )
    game_prediction_4_c1 = clients[1].update_prediction_with_beta(3, beta_game_2[1])
    # Check new predictions calculations
    assert torch.allclose(manual_game_prediction_4_c1, game_prediction_4_c1, rtol=0.0, atol=1e-5)

    # Put game client prediction in one Y matrix
    game_predictions_4 = torch.stack([game_prediction_4_c0, game_prediction_4_c1], dim=0).squeeze()
    # Compute regret with the game, we don't change mixture weights
    assert mixture_weights[3].shape == (NUM_CLIENTS, 1)
    assert game_predictions_4.shape == (NUM_CLIENTS, Y_DIM)

    sync_step_residual_game = (
        TARGET_SEQUENCE[4].unsqueeze(1) - torch.matmul(mixture_weights[3].T, game_predictions_4).T
    )
    residual_inner_product_game = torch.pow(torch.linalg.norm(sync_step_residual_game), 2.0)
    # Asserting that the game residual is smaller than the no game residual
    T_regularizer_c0_game = GAMMA * torch.pow(torch.linalg.norm(beta_game_2[0]), 2.0)
    T_regularizer_c1_game = GAMMA * torch.pow(torch.linalg.norm(beta_game_2[1]), 2.0)
    regret_game = 2 * residual_inner_product_game + T_regularizer_c0_game + T_regularizer_c1_game
    if check_game_regret:
        assert regret_no_game > regret_game
    if check_game_residual:
        assert residual_inner_product_no > residual_inner_product_game

    # Now let's see if previous beta optimized in the game is better than the one computed in the server (t=2)
    beta_game_2 = beta_game_2.reshape(NUM_CLIENTS, Z_DIM, 1)
    # We assume we are at step t = 2 and we use beta_1 to update the prediction
    # After debugging we found out we need to update previous prediction based on the game
    client_manager.improve_previous_predictions_from_game(2, past_game_predictions[-2], past_T_betas[-2])
    # client_0
    game_prediction_3_c0 = clients[0].update_prediction_with_beta(2, beta_game_2[0])
    # client_1
    game_prediction_3_c1 = clients[1].update_prediction_with_beta(2, beta_game_2[1])
    game_predictions_3 = torch.stack([game_prediction_3_c0, game_prediction_3_c1], dim=0).squeeze()
    step_2_residual_game = TARGET_SEQUENCE[3].unsqueeze(1) - torch.matmul(mixture_weights[2].T, game_predictions_3).T
    residual_inner_product_step2_game = torch.pow(torch.linalg.norm(step_2_residual_game), 2.0)
    # No game residual is 4.5550, and with game it decreases to 4.1021
    assert no_game_residuals[2] > residual_inner_product_step2_game

    # Now let's try t=1's game beta
    # If we use the following function, we don't need to reshape beta ourselves.
    # we use beta 0 to update the prediction at step 1
    clients_predictions_step1 = client_manager.get_predictions_with_beta(0, beta_game_1)
    step_1_residual_game = (
        TARGET_SEQUENCE[2].unsqueeze(1) - torch.matmul(mixture_weights[1].T, clients_predictions_step1).T
    )
    residual_inner_step_1_residual_game = torch.pow(torch.linalg.norm(step_1_residual_game), 2.0)
    assert no_game_residuals[1] > residual_inner_step_1_residual_game

    clients_predictions_step0 = client_manager.get_predictions_with_beta(0, beta_game_0)
    step_0_residual_game = (
        TARGET_SEQUENCE[1].unsqueeze(1) - torch.matmul(mixture_weights[0].T, clients_predictions_step0).T
    )
    residual_inner_step_0_residual_game = torch.pow(torch.linalg.norm(step_0_residual_game), 2.0)
    assert no_game_residuals[0] > residual_inner_step_0_residual_game

    # Question: what if I use game_beta_2 and mixture_weights_2 for step 2.
    game_clients_predictions_step3_beta2 = client_manager.get_predictions_with_beta(2, beta_game_2)
    step_2_residual_game = (
        TARGET_SEQUENCE[3].unsqueeze(1) - torch.matmul(mixture_weights[2].T, game_clients_predictions_step3_beta2).T
    )
    residual_inner_step_2_residual_game = torch.pow(torch.linalg.norm(step_2_residual_game), 2.0)
    # residual_inner_step_2_residual_game tensor(4.4808, grad_fn=<PowBackward0>)>)

    # It is better to use previous betas
    assert no_game_residuals[2] > residual_inner_step_2_residual_game
