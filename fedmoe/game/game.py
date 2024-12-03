from abc import ABC, abstractmethod
from typing import List

import torch

from fedmoe.clients.client import Client
from fedmoe.state.game_state import GameState

torch.set_default_dtype(torch.float64)


class Game(ABC):
    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        super().__init__()
        self.clients = clients
        self.num_clients = len(clients)
        self.sync_freq = sync_freq
        self.z_dim = z_dim
        self.y_dim = self.clients[0].y_dim
        assert self.z_dim == self.clients[0].z_dim, "Latent dimension for game and clients must match"
        self.current_time: int
        self.game_state = GameState()

    @abstractmethod
    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        raise NotImplementedError

    def get_A_hat_ij_t(self, time: int, i: int, j: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_D_client_ij_t(self, game_time: int, i: int, j: int, P_t_plus_1_client: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # TODO: check this function to make sure it is not changing the order of operations.
    # It is removed from RFN class.
    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        raise NotImplementedError

    def map_game_time_to_server_time(self, game_t: int, client: Client) -> int:
        """
        Maps the global time (current_time) in the server to the time scale used in game,
        that is between 0 to T, and returns the input associated with server time.

        Examples:

        For example, if the server time is 3T (self.current_time is a sync step) and the game time is t (0<=t<=T), we
        need to get the input at time 2T+t, so this function performs: 3T - T + t = 2T+t

        A numerical example: assume T = 4, and we are in the third round of synchronization,
        so self.current_time equals 12. In the game, we go from 12 to the previous sync step, which is 8,
        and need the model input during this time.
        When game time equals 3 (t=3), the input that we need is 8+3 based on the server time.
        This can be calculated by self.current_time (12) - self.sync_freq (4) + t (3) =  global time (11)

        """
        # Current time gives us the current sync step
        assert self.current_time == client.state.get_current_time()
        if self.current_time > self.sync_freq:
            return self.current_time - self.sync_freq + game_t
        return game_t

    def get_input(self, t: int, client: Client) -> torch.Tensor:
        """
        Maps the time t in the game (between 0 to sync_freq) to the time scale used in the server, current_time, and
        returns the input (x_t) associated with server time.
        """
        server_time = self.map_game_time_to_server_time(t, client)
        return client.get_x(server_time)

    def get_z(self, t: int, client: Client) -> torch.Tensor:
        """
        Maps the time t in the game (between 0 to sync_freq) to the time scale used in the server, current_time, and
        returns the hidden space value associated with server time.
        """
        server_time = self.map_game_time_to_server_time(t, client)
        return client.state.get_hidden_state_t(time=server_time)

    def get_e_alpha(self, t: int) -> torch.Tensor:
        """
        Calculates e^{- bold_alpha * (T- 1 - t)}
        """
        Iz = torch.eye(self.z_dim).double()
        alpha_tensor = torch.exp(
            torch.Tensor([-1 * client.alpha * (self.sync_freq - 1 - t) for client in self.clients])
        )
        bold_alpha = torch.block_diag(*[alpha * Iz for alpha in alpha_tensor])
        assert bold_alpha.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        return bold_alpha

    def get_e_alpha_gamma(self, t: int) -> torch.Tensor:
        """
        Calculates e^{- bold_alpha * (T- 1 - t)} * bold_gamma
        """
        e_alpha = self.get_e_alpha(t)
        Iz = torch.eye(self.z_dim).double()
        bold_gamma = torch.block_diag(*[client.gamma * Iz for client in self.clients])
        # Output shape: Nd_z x Nd_z
        assert bold_gamma.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        out = torch.matmul(e_alpha, bold_gamma)
        assert out.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        return out

    def create_bold_w_t(self, latest_mixture_weights: torch.Tensor) -> torch.Tensor:
        W_list = []
        for w_Tn in latest_mixture_weights:
            W_list.append(w_Tn.double() * torch.eye(self.y_dim).double())
        W_t = torch.cat(W_list, dim=1).T.double()
        assert W_t.shape == (self.num_clients * self.y_dim, self.y_dim)
        return W_t

    def first_block_alg2(self, time: int) -> None:
        """
        This function implements the first block in Algorithm 2.
        Latest mixture weights and y_T should correspond to the given time.
        """
        # Now that we are initializing P_T and S_T to zero, we don't need to pass latest_mixture_weights
        for client_id in range(0, self.num_clients):
            # P_t is N_d_y x Nd_y
            self.set_client_pt(
                t=time,
                client_id=client_id,
                pt_value=torch.zeros(self.num_clients * self.y_dim, self.num_clients * self.y_dim),
            )
            # S_t is N_d_y x 1
            self.set_client_st(t=time, client_id=client_id, st_value=torch.zeros(self.num_clients * self.y_dim, 1))

    def init_game_round_variables(self, current_time: int) -> None:
        # First update the local variable of clients
        self.current_time = current_time
        self.game_state.clear_state()
        self.game_state.init_state(self.sync_freq, self.num_clients, self.z_dim, self.y_dim)

        for client in self.clients:
            client.init_p_s(self.num_clients)
        self.e_alpha_gamma_A_inv = None

    def calculate_a(self, game_time: int) -> torch.Tensor:
        At = torch.zeros(self.num_clients, self.num_clients, self.z_dim, self.z_dim, dtype=torch.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_A_ij_t(game_time, i, j)
                At[i][j] = item
        # A_t shape: N*N*d_z*d_z
        return At

    def calculate_a_hat(self, game_time: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        assert bold_w_t.shape == (self.num_clients * self.y_dim, self.y_dim)
        A_hat_t = torch.zeros(self.num_clients, self.num_clients, self.z_dim, self.z_dim, dtype=torch.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_A_hat_ij_t(game_time, i, j, bold_w_t)
                A_hat_t[i][j] = item
        # A_hat_t shape: N*N*d_z*d_z
        return A_hat_t

    def calculate_b(self, t: int) -> torch.Tensor:
        B_list = []
        for client in self.clients:
            expected_e_Z = self.get_expectation_e_zt(t, client)
            # e is Ny_dim x y_dim and Z is y_dim x z_dim -> E[eZ]'s shape is Ny_dim x z_dim
            # P is Ny_dim x Ny_dim
            client_B = torch.matmul(client.P[t + 1].double(), expected_e_Z.double()).T
            assert client_B.shape == (self.z_dim, self.num_clients * self.y_dim)
            B_list.append(client_B)
        B = torch.cat(B_list, dim=0)
        assert B.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        return B

    def calculate_c(self, t: int) -> torch.Tensor:
        C_list = []
        for client in self.clients:
            expected_e_Z = self.get_expectation_e_zt(t, client)
            # E[eZ]'s shape is Ny_dim x z_dim
            # S_t's shape is Ny_dim x 1
            C_list.append(torch.matmul(expected_e_Z.T.double(), client.S[t + 1].double()))
        C = torch.cat(C_list, dim=0)
        assert C.shape == (self.num_clients * self.z_dim, 1)
        return C

    def calculate_d(self, t: int) -> torch.Tensor:
        D_list = []
        for client in self.clients:
            expected_e_Z = self.get_expectation_e_zt(t, client)
            D_list.append(expected_e_Z)
        # D's shape is Nd_y x Nd_z
        D = torch.cat(D_list, dim=1)
        assert D.shape == (self.num_clients * self.y_dim, self.num_clients * self.z_dim)
        return D

    def calculate_g(self, t: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        e_neg_alpha_t_gamma = self.get_e_alpha_gamma(t)
        e_neg_alpha_t = self.get_e_alpha(t)
        first_term = (
            e_neg_alpha_t_gamma
            + self.game_state.get_A_t(t)
            + torch.matmul(e_neg_alpha_t, self.game_state.get_A_hat_t(t))
        )
        w_tw_tT = torch.matmul(bold_w_t, bold_w_t.T)
        second_term = torch.matmul(
            torch.matmul(e_neg_alpha_t, self.game_state.get_D_t(t).T), w_tw_tT
        ) + self.game_state.get_B_t(t)

        g_t = torch.matmul(-1 * torch.linalg.inv(first_term), second_term)
        assert g_t.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        return g_t

    def calculate_h(self, t: int, bold_w_t: torch.Tensor, next_y: torch.Tensor) -> torch.Tensor:

        e_neg_alpha_t = self.get_e_alpha(t)
        e_neg_alpha_t_gamma = self.get_e_alpha_gamma(t).double()
        first_term = torch.add(
            torch.add(e_neg_alpha_t_gamma.double(), self.game_state.get_A_t(t).double()),
            torch.matmul(e_neg_alpha_t.double(), self.game_state.get_A_hat_t(t).double()),
        )

        assert first_term.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)

        bold_w_t_next_y = torch.matmul(bold_w_t.double(), next_y.double())
        assert next_y.shape == (self.y_dim, 1), f"Error in shape of next_y: {next_y.shape}"

        second_term = (
            torch.matmul(torch.matmul(e_neg_alpha_t, self.game_state.get_D_t(t).T.double()), bold_w_t_next_y.double())
            - self.game_state.get_C_t(t).double()
        )
        assert second_term.shape == (self.num_clients * self.z_dim, 1), f"shape is {second_term.shape}"
        inv_matrix = torch.linalg.inv(first_term)
        h_t = torch.matmul(inv_matrix, second_term.to(torch.float64))

        assert h_t.shape == (self.num_clients * self.z_dim, 1)
        return h_t

    def get_e_alpha_gamma_A_inv(self, t: int) -> torch.Tensor:
        A_t = self.game_state.get_A_t(t)
        e_alpha_gamma_t = self.get_e_alpha_gamma(t)  # output: Nd_z * Nd_z
        # output shape: N*d_z*N*d_z
        e_alpha_gamma_A = torch.add(e_alpha_gamma_t, A_t)
        assert e_alpha_gamma_A.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        return torch.linalg.inv(e_alpha_gamma_A).double()

    def calculate_Dt_client(self, game_time: int, client_id: int) -> torch.Tensor:
        Dt_client = torch.zeros(self.num_clients, self.num_clients, self.z_dim, self.z_dim, dtype=torch.float64)
        P_t_plus_1_client = self.get_client_pt(t=(game_time + 1), client_id=client_id)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_D_client_ij_t(game_time, i, j, P_t_plus_1_client)
                Dt_client[i][j] = item
        # Dt_client shape: N*N*d_z*d_z
        return Dt_client

    def calculate_pt_client(
        self,
        t: int,
        client_id: int,
        w_tw_tT: torch.Tensor,
    ) -> torch.Tensor:

        client_alpha = self.clients[client_id].alpha
        client_gamma = self.clients[client_id].gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        # bold_hat_e_i : shape [N*d_z, d_z]
        bold_hat_e_i = self.clients[client_id].get_hat_e(self.num_clients)

        e_client_alpha_t = torch.exp(torch.tensor(-1 * client_alpha) * (self.sync_freq - 1 - t))
        e_client_alpha_t_gamma = e_client_alpha_t * client_gamma

        line_1 = torch.matmul(
            torch.matmul(
                self.game_state.get_G_t(t).T,
                torch.add(
                    torch.add(self.game_state.get_A_hat_t(t), self.get_client_Dt(t, client_id)),
                    e_client_alpha_t_gamma * torch.matmul(bold_hat_e_i, bold_hat_e_i.T),
                ),
            ),
            self.game_state.get_G_t(t),
        )

        assert line_1.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)

        line_2 = torch.matmul(
            torch.matmul(((e_client_alpha_t * w_tw_tT) + p_next), self.game_state.get_D_t(t)),
            self.game_state.get_G_t(t),
        )
        assert line_2.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)

        line_3 = torch.matmul(
            torch.matmul(self.game_state.get_G_t(t).T, self.game_state.get_D_t(t).T),
            (e_client_alpha_t * w_tw_tT + p_next).T,
        )
        assert line_3.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)

        line_4 = e_client_alpha_t * w_tw_tT + p_next
        assert line_4.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)

        P_t = line_1 + line_2 + line_3 + line_4

        assert P_t.shape == (
            self.num_clients * self.y_dim,
            self.num_clients * self.y_dim,
        ), f"P(t)'s shape is {P_t.shape} but it should be (N*dy, N*dy)"
        # Shape should be (N*dy, N*dy)
        return P_t

    def calculate_st_client(
        self, t: int, client_id: int, w_tw_tT: torch.Tensor, wty_next_t: torch.Tensor
    ) -> torch.Tensor:
        client_alpha = self.clients[client_id].alpha
        client_gamma = self.clients[client_id].gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        s_next = self.get_client_st(t=(t + 1), client_id=client_id)
        e_client_alpha_t = torch.exp(torch.Tensor([(-1 * client_alpha) * (self.sync_freq - 1 - t)]))
        e_client_alpha_t_gamma = e_client_alpha_t * client_gamma
        # bold_hat_e_i : shape [N*d_z, d_z]
        bold_hat_e_i = self.clients[client_id].get_hat_e(self.num_clients)

        line_1 = torch.matmul(
            torch.matmul(
                self.game_state.get_G_t(t).T,
                self.game_state.get_A_hat_t(t)
                + self.get_client_Dt(t, client_id)
                + e_client_alpha_t_gamma * torch.matmul(bold_hat_e_i, bold_hat_e_i.T),
            ),
            self.game_state.get_H_t(t),
        )
        assert line_1.shape == (self.num_clients * self.y_dim, 1)

        line_2 = torch.matmul(
            torch.matmul(((e_client_alpha_t * w_tw_tT) + p_next), self.game_state.get_D_t(t)),
            self.game_state.get_H_t(t),
        )
        assert line_2.shape == (self.num_clients * self.y_dim, 1)

        line_3 = torch.matmul(
            torch.matmul(self.game_state.get_G_t(t).T, self.game_state.get_D_t(t).T),
            (-1 * wty_next_t * e_client_alpha_t + s_next),
        )
        assert line_3.shape == (self.num_clients * self.y_dim, 1)

        line_4 = -1 * wty_next_t * e_client_alpha_t + s_next
        assert line_4.shape == (self.num_clients * self.y_dim, 1)

        s_t = line_1 + line_2 + line_3 + line_4

        assert s_t.shape == (self.num_clients * self.y_dim, 1)
        return s_t

    def set_client_pt(self, t: int, client_id: int, pt_value: torch.Tensor) -> None:
        assert pt_value.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)
        assert (client_id, t) not in self.game_state._P_time_client_set
        self.clients[client_id].P[t] = pt_value
        # We record the set values
        self.game_state._P_time_client_set.add((client_id, t))

    def set_client_st(self, t: int, client_id: int, st_value: torch.Tensor) -> None:
        assert st_value.shape == (self.num_clients * self.y_dim, 1)
        assert (client_id, t) not in self.game_state._S_time_client_set
        self.clients[client_id].S[t] = st_value
        # We record the set values
        self.game_state._S_time_client_set.add((client_id, t))

    def set_client_Dt(self, game_time: int, client_id: int, dt_value: torch.Tensor) -> None:
        dt_value = dt_value.reshape(self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        assert (client_id, game_time) not in self.game_state._D_i_time_client_set
        self.clients[client_id].D[game_time] = dt_value
        self.game_state._D_i_time_client_set.add((client_id, game_time))

    def get_client_pt(self, t: int, client_id: int) -> torch.Tensor:
        assert (client_id, t) in self.game_state._P_time_client_set
        return self.clients[client_id].P[t]

    def get_client_st(self, t: int, client_id: int) -> torch.Tensor:
        assert (client_id, t) in self.game_state._S_time_client_set
        return self.clients[client_id].S[t]

    def get_client_Dt(self, game_time: int, client_id: int) -> torch.Tensor:
        assert (client_id, game_time) in self.game_state._D_i_time_client_set
        return self.clients[client_id].D[game_time]

    def compute_beta(self, t: int, past_predictions: torch.Tensor) -> torch.Tensor:
        assert past_predictions.shape == (self.num_clients * self.y_dim, 1)
        bold_beta = torch.matmul(self.game_state.get_G_t(t), past_predictions) + self.game_state.get_H_t(t)
        assert bold_beta.shape == (self.num_clients * self.z_dim, 1)
        return bold_beta

    def compute_z_beta_clients(self, t: int, bold_beta: torch.Tensor) -> torch.Tensor:
        z_beta_clients = []
        bold_beta = bold_beta.reshape(self.num_clients, self.z_dim, 1)
        for client_id in range(self.num_clients):
            # if t==3 and self.current_time==8:
            #     print("inside game client_id", client_id)
            #     print("inside game bold_beta", bold_beta[client_id])
            #     print("inside game z", self.get_z(t, self.clients[client_id]))
            z_beta_client = torch.matmul(self.get_z(t, self.clients[client_id]), bold_beta[client_id])
            z_beta_clients.append(z_beta_client)
        n_z_betas = torch.cat(z_beta_clients, dim=0)
        assert n_z_betas.shape == (self.num_clients * self.y_dim, 1)
        return n_z_betas
