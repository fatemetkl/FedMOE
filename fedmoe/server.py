from typing import List

import torch

from fedmoe.client_manager import ClientManager
from fedmoe.game import Game
from fedmoe.metrics import RMSEMetric


class Server:
    def __init__(self, sync_freq: int, client_manager: ClientManager, game: Game) -> None:
        self.sync_freq = sync_freq
        self.num_clients = client_manager.num_clients
        self.client_manager = client_manager
        self.game = game
        self.server_outputs: List[torch.Tensor]
        self.mixture_weights: List[torch.Tensor]
        self.observed_values: List[torch.Tensor]
        self.clients_predictions: List[torch.Tensor]
        self.d_z = self.client_manager.d_z
        self.rmse_metric = RMSEMetric()

        self.y_dim = self.client_manager.y_dim
        self.K: float = 0.1
        self.eta: int = 1

    def compute_mixture_weights(self, predictions: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        one_N = torch.ones(self.num_clients, 1).double()

        A = 2 * self.num_clients * torch.matmul(predictions.transpose(0, 1), predictions) + (
            self.K * torch.eye(self.num_clients)
        )
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
        w_t = torch.matmul(torch.inverse(A).double(), b.double() - (division.double() * one_N.double()))
        return w_t

    def sync_round(
        self,
        current_t: int,
        past_observed_values: List[torch.Tensor],
        past_mixture_weights: List[torch.Tensor],
        past_predictions: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        # T+1 last steps are considered
        # Server has the observed values of all clients for the past T time steps
        # Server has the mixture weights of all the clients for the past T time steps
        # Last time step (T)
        print("sync round", current_t)
        self.game.init_game_round_variables(past_mixture_weights[-1], past_observed_values[-1])

        for t in range(self.sync_freq - 1, -1, -1):
            A_t = None
            B_t = None
            C_t = None
            D_t = None
            # Shape = Nx1
            w_t = torch.tensor(
                [torch.matmul(w_Tn.double(), torch.eye(self.y_dim).double()) for w_Tn in past_mixture_weights[t]]
            ).reshape(self.num_clients * self.y_dim, self.y_dim)

            #  Compute A, B, C, and D based on the type of generative model
            A_t = self.game.calculate_a(t)
            self.game.set_A_t(t, A_t)
            B_t = self.game.calculate_b(t)
            self.game.set_B_t(t, B_t)
            C_t = self.game.calculate_c(t)
            self.game.set_C_t(t, C_t)
            D_t = self.game.calculate_d(t)
            self.game.set_D_t(t, D_t)

            # Parallel
            e_alpha_gamma_A_inv = self.game.get_e_alpha_gamma_A_inv(t)
            # e_alpha_gamma_A_inv shape: N*N*dz*dz
            e_alpha_gamma_A_inv = e_alpha_gamma_A_inv.reshape(
                (self.num_clients * self.d_z, self.num_clients * self.d_z)
            )

            initial_term = torch.matmul(w_t, w_t.transpose(0, 1))
            wtyt = torch.matmul(w_t.double(), past_observed_values[t])
            for client_id in range(0, self.num_clients):
                client_pt = self.game.calculate_pt_client(
                    t,
                    client_id,
                    e_alpha_gamma_A_inv.double(),
                    initial_term.double(),
                )
                self.game.set_client_pt(t, client_id, pt_value=client_pt)

                client_st = self.game.calculate_st_client(
                    t,
                    client_id,
                    e_alpha_gamma_A_inv.double(),
                    wtyt,
                )
                self.game.set_client_st(t, client_id, st_value=client_st)

        past_T_betas = []
        # 0 to T-1
        for t in range(0, self.sync_freq):
            beta_t = self.game.compute_beta(t, past_predictions)
            # Beta shape: Nd_z * d_y
            past_T_betas.append(beta_t)
        # New betas for t = 0 to T-1
        return past_T_betas

    def fit(self, num_rounds: int) -> None:
        for t in range(0, num_rounds):
            y_t = self.client_manager.get_y(t)
            y_t = y_t.reshape(self.y_dim, 1)
            # Store observed target values
            self.observed_values.append(y_t)

            # Compute predictions locally
            # Update Experts and return predictions
            predictions = self.client_manager.fit_clients(t)
            predictions = predictions.reshape(self.y_dim, self.num_clients)
            self.clients_predictions.append(predictions)

            # Server synchronize Local Expert predictions
            w_t = self.compute_mixture_weights(predictions, y_t)
            self.mixture_weights.append(w_t)

            # if t%T == 0, we improve predictions based on Nash game
            # Predictions are generated based on Nash game
            if t % self.sync_freq == 0 and t > 0:
                start_point = max(t - self.sync_freq, 0)
                #  Sending the past T observations (including the current t) and mixture weights for Nash game
                past_T_betas = self.sync_round(
                    t,
                    self.observed_values[start_point:t],
                    self.mixture_weights[start_point:t],
                    self.clients_predictions[start_point:t],
                )

                # Improve step Ts predictions with the new beta_T <-- beta_(T-1)
                improved_predictions = self.client_manager.get_predictions_with_beta(t, past_T_betas[-1])
                predictions = improved_predictions
                w_t = self.compute_mixture_weights(predictions.double(), y_t.double())
                self.mixture_weights[t] = w_t

            server_output = torch.matmul(predictions.double(), w_t.double())

            self.server_outputs.append(server_output)
            self.rmse_metric.update(server_output, y_t)

        # Compute metric
        print("Final metric value", self.rmse_metric.compute())
