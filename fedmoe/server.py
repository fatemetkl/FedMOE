from typing import Dict, List, Sequence

import torch
from fl4health.utils.metrics import Metric, MetricManager

from fedmoe.client_manager import ClientManager
from fedmoe.game import Game


class Server:

    def __init__(
        self,
        sync_freq: int,
        client_manager: ClientManager,
        game: Game,
        metrics: Sequence[Metric],
        kappa: float,
        eta: float,
    ) -> None:
        assert (
            sync_freq == client_manager.sync_freq
        ), "Sync Frequency of Server is not the same as Sync Frequency of Client Manager"
        assert sync_freq == game.sync_freq, "Sync Frequency of Server is not the same as the Sync Frequency of Game"
        assert client_manager.z_dim == game.z_dim, "Latent dimension of Client Manager is not the same as the Game"
        self.sync_freq = sync_freq
        self.num_clients = client_manager.num_clients
        self.y_dim = client_manager.y_dim
        self.client_manager = client_manager
        self.game = game
        self.server_outputs: List[torch.Tensor] = []
        self.mixture_weights: List[torch.Tensor] = []
        self.observed_values: List[torch.Tensor] = []
        self.clients_predictions: List[torch.Tensor] = []
        self.z_dim = self.client_manager.z_dim
        self.metrics = metrics
        self.metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="server")
        self.kappa: float = kappa
        self.eta: float = eta

    def compute_mixture_weights(self, predictions: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        # Size of predictions is d_y x N (corresponds to \mathbf{\hat{Y}}_t) and y_t is d_y x 1.

        assert predictions.shape == (self.y_dim, self.num_clients)
        assert y_t.shape == (self.y_dim, 1)

        one_N = torch.ones(self.num_clients, 1).double()

        A = 2 * (torch.matmul(predictions.transpose(0, 1), predictions) + self.kappa * torch.eye(self.num_clients))
        b = 2 * torch.transpose(torch.matmul(y_t.transpose(0, 1).double(), predictions.double()), 0, 1)

        numerator = (
            torch.matmul(
                torch.matmul(one_N.transpose(0, 1).double(), torch.inverse(A).double()),
                b.double(),
            )
            - self.eta
        )
        denominator = torch.matmul(
            torch.matmul(one_N.transpose(0, 1).double(), torch.inverse(A).double()),
            one_N.double(),
        )
        division = numerator.double() / denominator.double()
        w_t = torch.matmul(torch.inverse(A).double(), (b.double() - (division.double() * one_N.double())))

        return w_t

    def sync_round(
        self,
        current_t: int,
        past_observed_values: List[torch.Tensor],
        past_mixture_weights: List[torch.Tensor],
    ) -> List:
        # T last steps are considered
        # Server has the observed values of all clients for the past T time steps [0 to T] inclusive.
        # Server has the mixture weights of all the clients for the past T time steps [0 to T] inclusive.
        # Last time step (T)
        assert len(past_observed_values) == self.sync_freq + 1
        assert len(past_mixture_weights) == self.sync_freq + 1
        self.game.init_game_round_variables(current_t)
        # In first_block_alg2 S(T) and P(T) are initialized to zero.
        self.game.first_block_alg2(time=self.sync_freq)

        for game_t in range(self.sync_freq - 1, -1, -1):
            A_t = None
            B_t = None
            C_t = None
            D_t = None
            bold_w_t = self.game.create_bold_w_t(past_mixture_weights[game_t])
            #  Compute A, B, C, and D based on the type of the generative model
            A_t = self.game.calculate_a(game_t)
            self.game.game_state.set_A_t(game_t, A_t)
            A_hat_t = self.game.calculate_a_hat(game_t, bold_w_t)
            self.game.game_state.set_A_hat_t(game_t, A_hat_t)
            B_t = self.game.calculate_b(game_t)
            self.game.game_state.set_B_t(game_t, B_t)
            C_t = self.game.calculate_c(game_t)
            self.game.game_state.set_C_t(game_t, C_t)
            D_t = self.game.calculate_d(game_t)
            self.game.game_state.set_D_t(game_t, D_t)
            G_t = self.game.calculate_g(game_t, bold_w_t)
            self.game.game_state.set_G_t(game_t, G_t)
            H_t = self.game.calculate_h(game_t, bold_w_t, past_observed_values[game_t + 1])
            self.game.game_state.set_H_t(game_t, H_t)

            # Parallel
            w_tw_tT = torch.matmul(bold_w_t, bold_w_t.T)
            wty_next_t = torch.matmul(bold_w_t, past_observed_values[game_t + 1].double())
            for client_id in range(0, self.num_clients):
                client_Dt = self.game.calculate_Dt_client(game_t, client_id)
                self.game.set_client_Dt(game_t, client_id, dt_value=client_Dt)
                client_pt = self.game.calculate_pt_client(
                    game_t,
                    client_id,
                    w_tw_tT,
                )
                self.game.set_client_pt(game_t, client_id, pt_value=client_pt)

                client_st = self.game.calculate_st_client(
                    game_t,
                    client_id,
                    w_tw_tT,
                    wty_next_t,
                )
                self.game.set_client_st(game_t, client_id, st_value=client_st)

        past_T_betas = []
        game_improved_predictions = []
        # 0 to T-1
        # Observed value for all the clients is the same.
        Y_hat_0 = past_observed_values[0].repeat(self.num_clients, 1)
        game_improved_predictions.append(Y_hat_0)
        for t in range(0, self.sync_freq):
            beta_t = self.game.compute_beta(t, game_improved_predictions[t])
            z_beta = self.game.compute_z_beta_clients(t, beta_t)
            new_prediction = game_improved_predictions[t] + z_beta
            game_improved_predictions.append(new_prediction)
            # beta_t is a list of game calculated betas for each client.
            past_T_betas.append(beta_t)
        # New betas for t = 0 to T-1
        return past_T_betas

    def fit(self, num_rounds: int, have_sync: bool = True, update_last_Y_sync: bool = True) -> Dict[str, float]:
        self.metric_manager.clear()
        # We start from t = 1 instead of t = 0 zero.
        # Because to make predictions at step 0, we would need beta_0 which needs Y{-2} and Z{-2} that we don't have.

        # Initialized self.clients_predictions with Y_0^i in clients
        hat_Y_0 = self.client_manager.get_Y_0()
        self.clients_predictions.append(hat_Y_0)

        # Initialize w_neg1 randomly satisfying the constraint that the elements sum to eta.
        # We don't explicitly keep track of w_neg1,
        # but we use it to initialize first server prediction \hat{y}_0
        w_neg1 = torch.randn((self.num_clients, 1)).double()
        w_neg1 = self.eta * w_neg1 / torch.sum(w_neg1)
        # self.mixture_weights.append(w_neg1)
        # We don't need to calculate w_0 because it is computed in the first round.

        # Mixture weights produce our "server prediction" here: \hat{y}_0 = hat_Y_0 * w_neg1
        self.server_outputs.append(torch.matmul(hat_Y_0.double(), w_neg1))

        for t in range(0, num_rounds):
            # last_observed_value = y_{t} since we're predicting for t+1.
            last_observed_value = self.client_manager.get_y(t)
            # Store observed target values
            self.observed_values.append(last_observed_value)

            # Compute predictions locally
            # Update Experts and return predictions corresponding to time t+1 (\hat{Y}_{t+1}^i for every i in clients)
            new_predictions = self.client_manager.fit_clients(t)
            #  Now optimize mixture weights based on clients' previous time-step predictions (Y_{t}).
            #  We use t predictions because we also need ground truth (y_{t}).
            w_t = self.compute_mixture_weights(self.clients_predictions[t].double(), last_observed_value.double())
            self.mixture_weights.append(w_t)

            # if t%T == 0, we update predictions based on Nash game
            if have_sync and t % self.sync_freq == 0 and t > 0:
                start_point = max(t - self.sync_freq, 0)
                #  Sending the last T+1 observations and mixture weights for Nash game
                #  (last 0 to T) --> time[t-T-1, t] inclusive current sync step.
                assert len(self.mixture_weights) == len(self.observed_values)
                past_T_betas = self.sync_round(
                    t,
                    self.observed_values[start_point : t + 1],
                    self.mixture_weights[start_point : t + 1],
                )
                # Improve step Ts predictions with the new beta_T <-- beta_(T-1)
                new_predictions = self.client_manager.get_predictions_with_beta(t, past_T_betas[-1])

                # Optional: update past T predictions in each client
                # self.client_manager.update_past_predictions(t, past_T_betas)

                if update_last_Y_sync:
                    # Optional: improve previous client predictions (Y^i_{t-1})
                    # We can use Y^i_{t-1} in this round's mixture weight computation.
                    # Uses beta_{t-1} for updating predictions at t-1 (we used the same beta for updating Y_t)
                    self.clients_predictions[t - 1] = self.client_manager.get_predictions_with_beta(
                        t - 1, past_T_betas[-1]
                    )

            # A list of clients predictions is appended (Y_{t+1}^i for every i in N)
            self.clients_predictions.append(new_predictions)
            assert len(self.clients_predictions) == t + 2

            server_output = torch.matmul(new_predictions.double(), w_t.double())

            self.server_outputs.append(server_output)

            self.metric_manager.update({"server_predictions": server_output}, self.client_manager.get_y(t))

        # Compute metric
        final_metric_value = self.metric_manager.compute()
        return final_metric_value
