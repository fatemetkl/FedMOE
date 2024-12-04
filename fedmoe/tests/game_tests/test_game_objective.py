import torch

from fedmoe.game import TransformerGame
from fedmoe.server import Server
from fedmoe.tests.utils import get_data_and_target_sequences, get_transformer_client_manager

torch.set_default_dtype(torch.float64)
DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()

Z_DIM = 2
Y_DIM = 3  # This is fixed for this data
# T should be fixed in this test
T = 4
NUM_CLIENTS = 2
GAMMA = 1.0

DO_SYNC = True


def test_game_objective() -> None:
    """
    This function tests if the betas optimized in nash game are all better, and reduce residual.
    This is done by comparing residuals from the server and the ones created inside the game.
    The main purpose of the test is around visually confirming residual values, that gave the main insight
    to solve a issue which is we should use the latest Y in the game rather than the one previously saved in client.
    """
    # a lot of the notes are with manual seed 2
    torch.manual_seed(10)
    torch.set_default_dtype(torch.float64)

    client_manager = get_transformer_client_manager(Z_DIM, sync_freq=T, gamma=GAMMA, alpha=0.5)

    client_manager.clients[0].gamma = GAMMA
    client_manager.clients[1].gamma = GAMMA
    assert client_manager.clients[0].x_dim == 2
    # Having Y_0 initialized with y_0 helps the game improve.

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
    # Server steps

    # Initialization
    mixture_weights = []
    client_predictions = []
    server_outputs = []
    no_game_residuals = []
    game_residuals = []
    inside_game_predictions_residuals = []

    hat_Y_0 = client_manager.get_Y_0()
    client_predictions.append(hat_Y_0)

    w_neg1 = torch.randn((NUM_CLIENTS, 1)).double()
    w_neg1 = 1.0 * w_neg1 / torch.sum(w_neg1)
    server_outputs.append(torch.matmul(w_neg1.T, hat_Y_0.double()).T)

    # For every step 0 to T inclusive.
    for t in range(len(DATA_SEQUENCE) - 1):
        # at time t we are predicting t+1, so the last prediction is time 8 when we predict 9 (data length is 10).
        next_predictions = client_manager.fit_clients(t)
        client_predictions.append(next_predictions)
        w_t = server.compute_mixture_weights(
            client_predictions[t].double(), TARGET_SEQUENCE[t].reshape(-1, 1).double()
        )
        assert w_t.shape == (NUM_CLIENTS, 1)
        mixture_weights.append(w_t)
        regular_residual = TARGET_SEQUENCE[t + 1].unsqueeze(1) - torch.matmul(w_t.T, next_predictions).T
        regular_inner_residual = torch.pow(torch.linalg.norm(regular_residual), 2.0)
        no_game_residuals.append(regular_inner_residual)
        if t == 4 and DO_SYNC:
            # First game round
            past_T_betas, game_improved_predictions = server.sync_round(
                4,
                [
                    TARGET_SEQUENCE[0].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[1].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[2].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[3].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[4].reshape(-1, 1).double(),
                ],
                [
                    mixture_weights[0].double(),
                    mixture_weights[1].double(),
                    mixture_weights[2].double(),
                    mixture_weights[3].double(),
                    mixture_weights[4].double(),
                ],
            )
            # the output of the game is beta for 0 to T-1 (3)
            assert len(past_T_betas) == 4
            assert len(game_improved_predictions) == 5
            # the last item in game_improved_predictions is Y_T (here Y_4) and the first one is Y_0
            # The only thing that seems different between game Y and server Y is the order of operations.
            # In server we start from the last one (T) but in the game it goes from 0.
            # Step 3
            inside_game_pred_4 = game_improved_predictions[4].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_4 = (
                TARGET_SEQUENCE[4].unsqueeze(1) - torch.matmul(mixture_weights[3].T, inside_game_pred_4).T
            )
            inner_inside_game_residual_4 = torch.pow(torch.linalg.norm(inside_game_residual_4), 2.0)
            # Step 2
            inside_game_pred_3 = game_improved_predictions[3].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_3 = (
                TARGET_SEQUENCE[3].unsqueeze(1) - torch.matmul(mixture_weights[2].T, inside_game_pred_3).T
            )
            inner_inside_game_residual_3 = torch.pow(torch.linalg.norm(inside_game_residual_3), 2.0)
            # Step 1
            inside_game_pred_2 = game_improved_predictions[2].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_2 = (
                TARGET_SEQUENCE[2].unsqueeze(1) - torch.matmul(mixture_weights[1].T, inside_game_pred_2).T
            )
            inner_inside_game_residual_2 = torch.pow(torch.linalg.norm(inside_game_residual_2), 2.0)
            # Step 0

            inside_game_pred_1 = game_improved_predictions[1].reshape(NUM_CLIENTS, Y_DIM)
            # print("prediction 1 from inside game", inside_game_pred_1)
            inside_game_residual_1 = (
                TARGET_SEQUENCE[1].unsqueeze(1) - torch.matmul(mixture_weights[0].T, inside_game_pred_1).T
            )
            inner_inside_game_residual_1 = torch.pow(torch.linalg.norm(inside_game_residual_1), 2.0)

            inside_game_predictions_residuals.append(inner_inside_game_residual_1)
            inside_game_predictions_residuals.append(inner_inside_game_residual_2)
            inside_game_predictions_residuals.append(inner_inside_game_residual_3)
            inside_game_predictions_residuals.append(inner_inside_game_residual_4)

            # Computations after the game but in the server.
            # get new predictions for step T-4 (0)
            # we use beta 3 for prediction at step 4
            # important update the client prediction at t=4 with Y_4 in game
            for client in client_manager.clients:
                client.state.replace_prediction_t(
                    game_improved_predictions[4].reshape(NUM_CLIENTS, Y_DIM, 1)[client.id].reshape(-1, 1), 4
                )
            game_prediction_5 = client_manager.get_predictions_with_beta(t, past_T_betas[3])
            game_residual_5 = TARGET_SEQUENCE[5].unsqueeze(1) - torch.matmul(mixture_weights[4].T, game_prediction_5).T
            game_inner_residual_5 = torch.pow(torch.linalg.norm(game_residual_5), 2.0)

            # get new prediction for previous step T-1 (assuming we are at step 3 and we use the correct beta)
            game_prediction_4 = client_manager.get_predictions_with_beta(3, past_T_betas[3])
            game_residual_4 = TARGET_SEQUENCE[4].unsqueeze(1) - torch.matmul(mixture_weights[3].T, game_prediction_4).T
            game_inner_residual_4 = torch.pow(torch.linalg.norm(game_residual_4), 2.0)

            # get new predictions for step T-2 (2)
            game_prediction_3 = client_manager.get_predictions_with_beta(2, past_T_betas[2])
            game_residual_3 = TARGET_SEQUENCE[3].unsqueeze(1) - torch.matmul(mixture_weights[2].T, game_prediction_3).T
            game_inner_residual_3 = torch.pow(torch.linalg.norm(game_residual_3), 2.0)

            # get new predictions for step T-3 (1)
            game_prediction_2 = client_manager.get_predictions_with_beta(1, past_T_betas[1])
            game_residual_2 = TARGET_SEQUENCE[2].unsqueeze(1) - torch.matmul(mixture_weights[1].T, game_prediction_2).T
            game_inner_residual_2 = torch.pow(torch.linalg.norm(game_residual_2), 2.0)

            game_prediction_1 = client_manager.get_predictions_with_beta(0, past_T_betas[0])
            # print("game prediction 1 computed in server", game_prediction_1)
            game_residual_1 = TARGET_SEQUENCE[1].unsqueeze(1) - torch.matmul(mixture_weights[0].T, game_prediction_1).T
            game_inner_residual_1 = torch.pow(torch.linalg.norm(game_residual_1), 2.0)

            game_residuals.append(game_inner_residual_1)
            game_residuals.append(game_inner_residual_2)
            game_residuals.append(game_inner_residual_3)
            game_residuals.append(game_inner_residual_4)
            game_residuals.append(game_inner_residual_5)
            step_5_old_residual = game_inner_residual_5
            assert len(no_game_residuals) == len(game_residuals)

        elif t == 8 and DO_SYNC:
            # Second game round
            past_T_betas_2, game_improved_predictions = server.sync_round(
                8,
                [
                    TARGET_SEQUENCE[4].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[5].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[6].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[7].reshape(-1, 1).double(),
                    TARGET_SEQUENCE[8].reshape(-1, 1).double(),
                ],
                [
                    mixture_weights[4].double(),
                    mixture_weights[5].double(),
                    mixture_weights[6].double(),
                    mixture_weights[7].double(),
                    mixture_weights[8].double(),
                ],
            )
            # the output of the game is beta for 0 to T-1 (3)
            assert len(past_T_betas_2) == 4
            # last item in game_improved_predictions is Y_T, in this case it would be Y_8
            assert len(game_improved_predictions) == 5

            # Step 7
            inside_game_pred_8 = game_improved_predictions[4].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_8 = (
                TARGET_SEQUENCE[8].unsqueeze(1) - torch.matmul(mixture_weights[7].T, inside_game_pred_8).T
            )
            inner_inside_game_residual_8 = torch.pow(torch.linalg.norm(inside_game_residual_8), 2.0)
            # Step 6
            inside_game_pred_7 = game_improved_predictions[3].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_7 = (
                TARGET_SEQUENCE[7].unsqueeze(1) - torch.matmul(mixture_weights[6].T, inside_game_pred_7).T
            )
            inner_inside_game_residual_7 = torch.pow(torch.linalg.norm(inside_game_residual_7), 2.0)
            # Step 5
            inside_game_pred_6 = game_improved_predictions[2].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_6 = (
                TARGET_SEQUENCE[6].unsqueeze(1) - torch.matmul(mixture_weights[5].T, inside_game_pred_6).T
            )
            inner_inside_game_residual_6 = torch.pow(torch.linalg.norm(inside_game_residual_6), 2.0)
            # Step 4
            inside_game_pred_5 = game_improved_predictions[1].reshape(NUM_CLIENTS, Y_DIM)
            inside_game_residual_5 = (
                TARGET_SEQUENCE[5].unsqueeze(1) - torch.matmul(mixture_weights[4].T, inside_game_pred_5).T
            )
            inner_inside_game_residual_5 = torch.pow(torch.linalg.norm(inside_game_residual_5), 2.0)

            inside_game_predictions_residuals.append(inner_inside_game_residual_5)
            inside_game_predictions_residuals.append(inner_inside_game_residual_6)
            inside_game_predictions_residuals.append(inner_inside_game_residual_7)
            inside_game_predictions_residuals.append(inner_inside_game_residual_8)

            # we use beta 7 for prediction at step 8 (predicting for 9)
            game_prediction_9 = client_manager.get_predictions_with_beta(8, past_T_betas_2[3])
            game_residual_9 = TARGET_SEQUENCE[9].unsqueeze(1) - torch.matmul(mixture_weights[8].T, game_prediction_9).T
            game_inner_residual_9 = torch.pow(torch.linalg.norm(game_residual_9), 2.0)
            # get new prediction for previous step T-1 (assuming we are at step 7 and we use the correct beta)
            game_prediction_8 = client_manager.get_predictions_with_beta(7, past_T_betas_2[3])
            game_residual_8 = TARGET_SEQUENCE[8].unsqueeze(1) - torch.matmul(mixture_weights[7].T, game_prediction_8).T
            game_inner_residual_8 = torch.pow(torch.linalg.norm(game_residual_8), 2.0)
            # get new predictions for step T-2 (6)
            game_prediction_7 = client_manager.get_predictions_with_beta(6, past_T_betas_2[2])
            game_residual_7 = TARGET_SEQUENCE[7].unsqueeze(1) - torch.matmul(mixture_weights[6].T, game_prediction_7).T
            game_inner_residual_7 = torch.pow(torch.linalg.norm(game_residual_7), 2.0)
            # get new predictions for step T-3 (5)
            game_prediction_6 = client_manager.get_predictions_with_beta(5, past_T_betas_2[1])
            game_residual_6 = TARGET_SEQUENCE[6].unsqueeze(1) - torch.matmul(mixture_weights[5].T, game_prediction_6).T
            game_inner_residual_6 = torch.pow(torch.linalg.norm(game_residual_6), 2.0)
            # get new predictions for step T-4 (4)
            game_prediction_5 = client_manager.get_predictions_with_beta(4, past_T_betas_2[0])
            game_residual_5 = TARGET_SEQUENCE[5].unsqueeze(1) - torch.matmul(mixture_weights[4].T, game_prediction_5).T
            game_inner_residual_5 = torch.pow(torch.linalg.norm(game_residual_5), 2.0)
            # game_residuals.append(game_inner_residual_5)
            step_5_new_residual = game_inner_residual_5
            game_residuals.append(game_inner_residual_6)
            game_residuals.append(game_inner_residual_7)
            game_residuals.append(game_inner_residual_8)
            game_residuals.append(game_inner_residual_9)

            assert len(no_game_residuals) == len(
                game_residuals
            ), f"Error no_game_residuals {len(no_game_residuals)}, game_residuals {len(game_residuals)}"

            # to make current step better (predict for 9), we can use the games Y_8 in its prediction
            for client in client_manager.clients:
                client.state.replace_prediction_t(
                    game_improved_predictions[4].reshape(NUM_CLIENTS, Y_DIM, 1)[client.id].reshape(-1, 1), 8
                )
            # now do the regular thing in the server
            game_prediction_9_new = client_manager.get_predictions_with_beta(8, past_T_betas_2[3])
            game_residual_9_new = (
                TARGET_SEQUENCE[9].unsqueeze(1) - torch.matmul(mixture_weights[8].T, game_prediction_9_new).T
            )
            game_residual_inner_9_new = torch.pow(torch.linalg.norm(game_residual_9_new), 2.0)

    if DO_SYNC:
        for game_res, no_game, time in zip(game_residuals, no_game_residuals, range(1, len(game_residuals) + 1)):
            print(f"no game {no_game},  server game {game_res},  step {time-1} predicting for {time}")
            if time == 5:
                print(
                    f"time is 4 (T): step 5 new game residual,{step_5_new_residual}",
                    f"vs old game residual {step_5_old_residual}",
                )
        for time, in_game, no_game in zip(
            range(0, len(game_residuals)), inside_game_predictions_residuals, no_game_residuals
        ):
            assert no_game >= in_game
            print(f"inside game residual {in_game}, time {time} predicting {time+1}")

        print("This is what we get by using game Y_8: step 8 predicting for 9: ", game_residual_inner_9_new)

    else:
        assert len(no_game_residuals) == 9
        print("we have not played the game in this case:")
        for time, no_game in enumerate(no_game_residuals):
            print(f"no game {no_game},  step {time} predicting for {time+1}")

    # Now try the whole server and see if it also uses the game prediction of Ts correctly.
    # Reset everything here
    # torch.manual_seed(10)
    # client_manager = get_transformer_client_manager(Z_DIM, sync_freq=T, gamma=GAMMA, alpha=0.5)

    # game = TransformerGame(
    #     client_manager.clients,
    #     sync_freq=T,
    #     z_dim=Z_DIM,
    # )

    # server2 = Server(
    #     total_game_steps=T,
    #     client_manager=client_manager,
    #     game=game,
    #     metrics=[],
    #     kappa=1.0,
    #     eta=1.0,
    # )
    # server_residuals = []

    # # server goes from 0 to round-1
    # _ = server2.fit(9, have_sync=True)
    # # calculate residuals
    # for t in range(10):
    #     w_t = server2.mixture_weights[t]
    #     next_predictions = server2.clients_predictions[t + 1]
    #     residual = TARGET_SEQUENCE[t + 1].unsqueeze(1) - torch.matmul(w_t.T, next_predictions).T
    #     inner_residual = torch.pow(torch.linalg.norm(residual), 2.0)
    #     server_residuals.append(inner_residual)
    #     print(f"server residual {inner_residual}, time {t} predicting {t+1}")

    # assert False
