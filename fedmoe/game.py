import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from fedmoe.clients.client import Client
from fedmoe.clients.esn_client import EchoStateNetworkClient


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

    @abstractmethod
    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        raise NotImplementedError

    # TODO: check this function to make sure it is not changing the order of operations.
    # It is removed from RFN class.
    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        raise NotImplementedError

    def init_loop_times(self) -> None:
        self._A_times_set: set[int] = set()
        self._B_times_set: set[int] = set()
        self._C_times_set: set[int] = set()
        self._D_times_set: set[int] = set()
        self._S_time_client_set: set[Tuple[int, int]] = set()
        self._P_time_client_set: set[Tuple[int, int]] = set()

    def map_game_time_to_server_time(self, t: int, client: Client) -> int:
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
            return self.current_time - self.sync_freq + t
        return t

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

    def get_e_alpha_gamma(self, t: int) -> torch.Tensor:
        Iz = torch.eye(self.z_dim).double()
        bold_alpha = torch.block_diag(*[client.alpha * Iz for client in self.clients])
        bold_gamma = torch.block_diag(*[client.gamma * Iz for client in self.clients])
        e_alpha = torch.linalg.matrix_exp(-1 * bold_alpha * (self.sync_freq - t))
        # Output shape: Nd_z x Nd_z
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

    def first_block_alg2(self, latest_mixture_weights: torch.Tensor, latest_y: torch.Tensor, time: int) -> None:
        """
        This function implements the first block in Algorithm 2.
        Latest mixture weights and y_T should correspond to the given time.
        """
        assert latest_mixture_weights.shape == (self.num_clients, 1)
        assert latest_y.shape == (self.y_dim, 1)

        W_current = self.create_bold_w_t(latest_mixture_weights)

        for client_id in range(0, self.num_clients):
            # P_t is N_d_y x Nd_y
            self.set_client_pt(t=time, client_id=client_id, pt_value=(torch.matmul(W_current, W_current.T)))
            self.set_client_st(t=time, client_id=client_id, st_value=torch.matmul(-1 * W_current, latest_y.double()))

    def init_game_round_variables(self, current_time: int) -> None:
        # First update the local variable of clients
        self.current_time = current_time
        self.init_loop_times()

        for client in self.clients:
            client.init_p_s(self.num_clients)
        self.e_alpha_gamma_A_inv = None

        # Go over past time steps and record these values for t in [0, T-1]
        self.A = torch.zeros(
            (self.sync_freq, self.num_clients * self.z_dim, self.num_clients * self.z_dim),
            dtype=torch.float64,
        )
        self.B = torch.zeros(
            (self.sync_freq, self.num_clients * self.z_dim, self.num_clients * self.y_dim),
            dtype=torch.float64,
        )
        self.C = torch.zeros(
            (self.sync_freq, self.num_clients * self.z_dim, 1),
            dtype=torch.float64,
        )
        self.D = torch.zeros(
            (self.sync_freq, self.num_clients * self.y_dim, self.num_clients * self.z_dim),
            dtype=torch.float64,
        )

    def calculate_a(self, time: int) -> torch.Tensor:
        At = torch.zeros(self.num_clients, self.num_clients, self.z_dim, self.z_dim, dtype=torch.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_A_ij_t(time, i, j)
                At[i][j] = item
        # A_t shape: N*N*d_z*d_z
        return At

    def calculate_b(self, t: int) -> torch.Tensor:
        B_list = []
        for client in self.clients:
            expected_e_Z = self.get_expectation_e_zt(t, client)
            # e is Ny_dim x y_dim and Z is y_dim x z_dim -> E[eZ]'s shape is Ny_dim x z_dim
            # P is Ny_dim x Ny_dim
            client_B = torch.matmul(expected_e_Z.T.double(), client.P[t + 1].double())
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

    def set_A_t(self, t: int, A_t: torch.Tensor) -> None:
        self.A[t] = A_t.reshape(self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        assert t not in self._A_times_set
        self._A_times_set.add(t)

    def get_A_t(self, t: int) -> torch.Tensor:
        assert t in self._A_times_set
        return self.A[t]

    def set_B_t(self, t: int, B_t: torch.Tensor) -> None:
        assert B_t.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        self.B[t] = B_t
        assert t not in self._B_times_set
        self._B_times_set.add(t)

    def get_B_t(self, t: int) -> torch.Tensor:
        assert t in self._B_times_set
        return self.B[t]

    def set_C_t(self, t: int, C_t: torch.Tensor) -> None:
        assert C_t.shape == (self.num_clients * self.z_dim, 1)
        self.C[t] = C_t
        assert t not in self._C_times_set
        self._C_times_set.add(t)

    def get_C_t(self, t: int) -> torch.Tensor:
        assert t in self._C_times_set
        return self.C[t]

    def set_D_t(self, t: int, D_t: torch.Tensor) -> None:
        assert D_t.shape == (self.num_clients * self.y_dim, self.num_clients * self.z_dim)
        self.D[t] = D_t
        assert t not in self._D_times_set
        self._D_times_set.add(t)

    def get_D_t(self, t: int) -> torch.Tensor:
        assert t in self._D_times_set
        return self.D[t]

    def get_e_alpha_gamma_A_inv(self, t: int) -> torch.Tensor:
        A_t = self.get_A_t(t)
        e_alpha_gamma_t = self.get_e_alpha_gamma(t)  # output: Nd_z * Nd_z
        # output shape: N*d_z*N*d_z
        e_alpha_gamma_A = torch.add(e_alpha_gamma_t, A_t)
        assert e_alpha_gamma_A.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        return torch.inverse(e_alpha_gamma_A).double()

    def calculate_pt_client(
        self,
        t: int,
        client_id: int,
        e_alpha_gamma_A_inv: torch.Tensor,
        initial_term: torch.Tensor,
    ) -> torch.Tensor:

        client_alpha = self.clients[client_id].alpha
        client_gamma = self.clients[client_id].gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        # e_i : shape [N*d_y, d_y]
        e_client_alpha_t = torch.exp(torch.tensor(-1 * client_alpha) * (self.sync_freq - t))
        e_client_alpha_t_gamma = e_client_alpha_t * client_gamma

        I_matrix = torch.eye(self.num_clients * self.y_dim).double()

        x = torch.matmul(self.get_D_t(t).double(), e_alpha_gamma_A_inv)
        m = torch.matmul(x.double(), self.get_B_t(t).double())
        term_1_pre = (m - I_matrix).transpose(0, 1)

        # term_1 shape: N*N
        term_1 = torch.matmul(term_1_pre, p_next)
        term_2 = (
            torch.matmul(torch.matmul(self.get_D_t(t), e_alpha_gamma_A_inv).double(), self.get_B_t(t).double())
            - I_matrix
        )
        # term_2 shape: N*N
        term_1_2 = torch.matmul(term_1, term_2)

        term_3_part2 = torch.matmul(e_alpha_gamma_A_inv, self.get_B_t(t).double())
        term_3_part1 = e_client_alpha_t_gamma * term_3_part2.transpose(0, 1)
        term_3 = torch.matmul(term_3_part1, term_3_part2)
        term_4 = e_client_alpha_t * initial_term
        P_t = torch.add(torch.add(term_1_2, term_3), term_4)
        assert P_t.shape == (
            self.num_clients * self.y_dim,
            self.num_clients * self.y_dim,
        ), f"P(t)'s shape is {P_t.shape} but it should be (N*dy, N*dy)"
        # Shape should be (N*dy, N*dy)
        return P_t

    def calculate_st_client(
        self, t: int, client_id: int, e_alpha_gamma_A_inv: torch.Tensor, wtyt: torch.Tensor
    ) -> torch.Tensor:
        client_alpha = self.clients[client_id].alpha
        client_gamma = self.clients[client_id].gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        s_next = self.get_client_st(t=(t + 1), client_id=client_id)
        e_client_alpha_t = torch.exp(torch.Tensor([(-1 * client_alpha) * (self.sync_freq - t)]))
        I_matrix = torch.eye(self.num_clients * self.z_dim).double()

        term_1_1 = torch.matmul(self.get_B_t(t).T, e_alpha_gamma_A_inv)
        DPD = torch.matmul(torch.matmul(self.get_D_t(t).T, p_next), self.get_D_t(t))
        term_1_2 = torch.add(e_client_alpha_t * client_gamma * I_matrix, DPD)
        term_1_3 = torch.matmul(e_alpha_gamma_A_inv, self.get_C_t(t))
        term_2 = torch.matmul(
            torch.matmul(torch.matmul(p_next, self.D[t]), e_alpha_gamma_A_inv),
            self.get_C_t(t),
        )
        term_3 = torch.matmul(
            torch.matmul(
                torch.matmul(self.get_B_t(t).T, e_alpha_gamma_A_inv),
                self.get_D_t(t).T,
            ),
            s_next,
        )
        s_term = torch.matmul(torch.matmul(term_1_1, term_1_2), term_1_3) - term_2 - term_3
        s_term = s_term + s_next - e_client_alpha_t * wtyt
        assert s_term.shape == (self.num_clients * self.y_dim, 1)
        return s_term

    def set_client_pt(self, t: int, client_id: int, pt_value: torch.Tensor) -> None:
        assert pt_value.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)
        assert (client_id, t) not in self._P_time_client_set
        self.clients[client_id].P[t] = pt_value
        # We record the set values
        self._P_time_client_set.add((client_id, t))

    def set_client_st(self, t: int, client_id: int, st_value: torch.Tensor) -> None:
        assert st_value.shape == (self.num_clients * self.y_dim, 1)
        assert (client_id, t) not in self._S_time_client_set
        self.clients[client_id].S[t] = st_value
        # We record the set values
        self._S_time_client_set.add((client_id, t))

    def get_client_pt(self, t: int, client_id: int) -> torch.Tensor:
        assert (client_id, t) in self._P_time_client_set
        return self.clients[client_id].P[t]

    def get_client_st(self, t: int, client_id: int) -> torch.Tensor:
        assert (client_id, t) in self._S_time_client_set
        return self.clients[client_id].S[t]

    def compute_beta(self, t: int, past_predictions: torch.Tensor) -> torch.Tensor:
        # past_predictions is dy x N and B[t] is N*d_z x N*d_y
        # So we should change the shape of Y to N*d_y x 1, where we are stacking each row on top of each other.
        # Transpose first changes the shape from dy x N to N x d_y, then we apply reshape.
        # Note that in the paper bold \hat{Y} has shape d_y x N, but non-bold \hat{Y} is Nd_y x 1.
        past_predictions = past_predictions.T.reshape(-1, 1)
        e_alpha_gamma_A_inv = self.get_e_alpha_gamma_A_inv(t)
        assert e_alpha_gamma_A_inv.shape == (self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        bold_beta = torch.matmul(
            -1 * e_alpha_gamma_A_inv.double(),
            torch.add(
                torch.matmul(self.get_B_t(t).double(), past_predictions.double()),
                self.get_C_t(t).double(),
            ),
        )
        assert bold_beta.shape == (self.num_clients * self.z_dim, 1)
        # bold_beta shape is Nd_z x 1
        # To be able to get beta for each client more easily, we reshape bold beta
        bold_beta = bold_beta.reshape(-1, self.z_dim, 1)
        return bold_beta


class TransformerGame(Game):
    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        super().__init__(clients, sync_freq, z_dim)

    def get_input(self, t: int, client: Client) -> torch.Tensor:
        """
        Maps the time t in the game (between 0 to sync_freq) to the time scale used in the server, current_time, and
        returns the input (x_t) associated with server time.
        """
        server_time = self.map_game_time_to_server_time(t, client)
        # Assuming that the input shape in transformer is (x_dim, 1)
        return client.get_x(server_time)

    def get_expectation_e_zt(self, t: int, client: Client) -> torch.Tensor:
        Z = client.feed_encoder(self.get_input(t, client).double())
        # Embedding shape is y_dim x z_dim
        assert Z.shape == (self.y_dim, self.z_dim)
        # e_i's shape is (num_clients * self.y_dim, self.y_dim)
        e_i = client.get_e(self.num_clients)
        # output shape is Ny_dim x z_dim
        return torch.matmul(
            e_i.double(),
            Z.double(),
        )

    def get_A_ij_t(self, t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_E = self.get_expectation_e_zt(t, client_i)
        client_j_E = self.get_expectation_e_zt(t, client_j)
        return torch.matmul(torch.matmul(client_i_E.T, client_i.P[t + 1]), client_j_E)


class EchoStateGame(Game):
    def __init__(
        self,
        clients: List[Client],
        sync_freq: int,
        z_dim: int,
        N_samples: int = 100,
    ) -> None:
        super().__init__(clients, sync_freq, z_dim)
        self.N_samples = N_samples
        for client in clients:
            assert client.sync_steps == sync_freq

    def get_input(self, t: int, client: Client) -> torch.Tensor:
        """
        Maps the time t in the game (between 0 to sync_freq) to the time scale used in the server, current_time, and
        returns the input (x_t) associated with server time.
        """
        server_time = self.map_game_time_to_server_time(t, client)
        return client.get_input_matrix(server_time)

    def simulate_z_t(self, t: int, client: Client) -> torch.Tensor:
        # Setting z start, which is the last z before the last sync step.
        # Based on the game time scale, it is t=-1.
        Z = self.get_z(t=-1, client=client)
        #  Starts the simulation from -1 (last sync step -1) to desired t
        for back_t in range(0, t + 1):
            Z = client.encoder(self.get_input(back_t, client), Z, client.sigma)
        return Z

    def get_expectation_e_zt(self, t: int, client: Client) -> torch.Tensor:
        # Note that t here is not server time, but rather game time [0, sync_freq]
        assert type(client) is EchoStateNetworkClient
        samples = []
        for _ in range(self.N_samples):
            # Start from time current_t-T-1 for the trajectory
            # If we're PREDICTING t=8, with sync frequency T=4, then we want to generate trajectories
            # Z_3 -> Z_4 -> Z_5 -> Z_6 for our Nash game.
            # This means we start from Z_3 and use x_4, x_5, x_6 to generate these latent values.
            # To get Z_5, we again start from Z_3 -> Z_4 -> z_5
            estimated_Z_t = self.simulate_z_t(t, client)
            e_i = client.get_e(self.num_clients)
            e_z_T = torch.matmul(e_i.double(), estimated_Z_t.double())
            samples.append(e_z_T)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor = sum_tensor + sample
        return sum_tensor / self.N_samples

    def get_expectation_z_t_e_t_transpose_P_z_t_e_t(self, t: int, client: Client) -> torch.Tensor:
        # Note that t here is not server time, but rather game time [0, sync_freq]
        # When the z_t values are the same in these expectation calculations, we can't compute the expectations
        # separately. So we do everything together.
        assert type(client) is EchoStateNetworkClient
        samples = []
        for _ in range(self.N_samples):
            # Start from time current_t-T-1 for the trajectory
            # If we're PREDICTING t=8, with sync frequency T=4, then we want to generate trajectories
            # Z_3 -> Z_4 -> Z_5 -> Z_6 for our Nash game.
            # This means we start from Z_3 and use x_4, x_5, x_6 to generate these latent values.
            Z = self.simulate_z_t(t, client)
            e_i = client.get_e(self.num_clients)
            e_z_T = torch.matmul(e_i.double(), Z.double())
            P_t_plus_1 = client.P[t + 1]
            sample = torch.matmul(torch.matmul(e_z_T.T, P_t_plus_1), e_z_T)
            samples.append(sample)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor = sum_tensor + sample
        return sum_tensor / self.N_samples

    def get_A_ij_t(self, t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        if i != j:
            # If we're in the off diagonal entries of A_{ij}(t) then the expectations are independent
            # So we can calculate everything separately
            client_i_E = self.get_expectation_e_zt(t, client_i)
            client_j_E = self.get_expectation_e_zt(t, client_j)
            return torch.matmul(torch.matmul(client_i_E.T, client_i.P[t + 1]), client_j_E)
        else:
            # If we're on the diagonal, then the Z_ts are not independent and we have to do an estimation of everything
            # combined
            return self.get_expectation_z_t_e_t_transpose_P_z_t_e_t(t, client_i)


class RfnGame(Game):
    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        super().__init__(clients, sync_freq, z_dim)

    def get_input(self, t: int, client: Client) -> torch.Tensor:
        """
        Maps the time t in the game (between 0 to sync_freq) to the time scale used in the server, current_time, and
        returns the input (x_t) associated with server time.
        """
        server_time = self.map_game_time_to_server_time(t, client)
        return client.get_input_matrix(server_time)

    def get_a_t_embedding(self, time: int, client: Client) -> torch.Tensor:
        # some parts of the forwards pass (without randomness)
        a_t = (torch.matmul(client.encoder.A.double(), self.get_input(time, client).double())) + client.encoder.b
        assert a_t.shape == (self.y_dim, self.z_dim)
        return a_t

    def get_expectation_zt(self, time: int, client: Client) -> torch.Tensor:
        a_t = self.get_a_t_embedding(time, client)
        phi = torch.distributions.Normal(0, 1).cdf((torch.mul((-1 / client.sigma), a_t)))
        one_bar = torch.ones(client.z_dim)
        exp_term = torch.exp((-1 / (2 * (client.sigma) ** 2)) * torch.mul(a_t, a_t))
        second_term = client.sigma / math.sqrt(2 * math.pi) * exp_term
        a_ti = torch.mul(a_t, (one_bar - phi)) + second_term
        assert a_ti.shape == (self.y_dim, self.z_dim)
        return a_ti

    def calculate_b(self, t: int) -> torch.Tensor:
        B = []
        for client in self.clients:
            a_ti = self.get_expectation_zt(t, client)  # shape: 1*d_z
            e_i = client.get_e(self.num_clients)  # shape: Nd_y*d_y
            B.append(
                torch.matmul(
                    torch.matmul(self.get_client_pt(t=(t + 1), client_id=client.id).double(), e_i.double()), a_ti
                )
            )
        B_matrix = torch.cat(B, dim=1).T
        assert B_matrix.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        return B_matrix

    def calculate_c(self, t: int) -> torch.Tensor:
        C = []
        for client in self.clients:
            a_ti = self.get_expectation_zt(t, client)  # shape: 1*d_z
            e_i = client.get_e(self.num_clients)  # shape: Nd_y*d_y
            C.append(
                torch.matmul(
                    torch.matmul(self.get_client_st(t=(t + 1), client_id=client.id).T.double(), e_i.double()), a_ti
                )
            )
        C_matrix = torch.cat(C, dim=1).T
        assert C_matrix.shape == (self.num_clients * self.z_dim, 1)
        return C_matrix

    def calculate_d(self, t: int) -> torch.Tensor:
        D_list = []
        for client in self.clients:
            a_ti = self.get_expectation_zt(t, client)  # shape: 1*d_z
            e_i = client.get_e(self.num_clients)  # shape: Nd_y*d_y
            D_list.append(torch.matmul(e_i, a_ti))
        # D's shape is Nd_y x Nd_z
        D = torch.cat(D_list, dim=1)
        assert D.shape == (self.num_clients * self.y_dim, self.num_clients * self.z_dim)
        return D

    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        # Check that client_i.P[time + 1] is set
        next_p = self.get_client_pt(t=(time + 1), client_id=i).reshape(
            (self.num_clients, self.num_clients, self.y_dim, self.y_dim)
        )
        # a_ti is a(t,i) which is the E[Z_t^i] for RFNs; shape: d_y*d_z
        # at is a partial forward pass without randomness used in a(t,i) calculations. shape: d_y*d_Z
        if i != j:
            a_ti = self.get_expectation_zt(time, client_i)
            a_tj = self.get_expectation_zt(time, client_j)
            A_ij = torch.matmul((a_ti.T * next_p[i][j]), a_tj)
            # A_ij shape: d_z*d_z
            assert A_ij.shape == (self.z_dim, self.z_dim)
            return A_ij
        else:
            A_ii = torch.zeros(client_i.z_dim, client_i.z_dim, dtype=torch.float64)
            a_ti = self.get_expectation_zt(time, client_i)
            at = self.get_a_t_embedding(time, client_i)

            for p in range(client_i.z_dim):
                for k in range(client_i.z_dim):
                    if p == k:
                        phi = torch.distributions.Normal(0, 1).cdf(-at[:, p] / client_i.sigma)

                        second_term = (torch.mul(at, at)[:, p] + client_i.sigma**2) * (1 - phi) + at[
                            :, p
                        ] * client_i.sigma / math.sqrt(2 * math.pi) * torch.exp(
                            -1 * (torch.mul(at, at)[:, p]) / (2 * (client_i.sigma**2))
                        )
                    else:
                        second_term = a_ti[:, p] * a_ti[:, k]
                    A_ii[p][k] = next_p[i][i] * second_term
            # A_ii shape: d_z*d_z
            assert A_ii.shape == (self.z_dim, self.z_dim)
            return A_ii
