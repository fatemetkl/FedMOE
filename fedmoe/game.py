import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from fedmoe.clients.client import Client
from fedmoe.clients.esn_client import EchoStateNetworkClient


class Game(ABC):
    def __init__(self, clients: List[Client], sync_freq: int, d_z: int) -> None:
        super().__init__()
        self.clients = clients
        self.num_clients = len(clients)
        self.sync_freq = sync_freq
        self.d_z = d_z
        self.y_dim = self.clients[0].y_dim

    @abstractmethod
    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        raise NotImplementedError

    # @abstractmethod
    # TODO: this function should be removed because it might be changing the order of operations
    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        raise NotImplementedError

    def init_loop_times(self) -> None:
        self._A_times_set: set[int] = set()
        self._B_times_set: set[int] = set()
        self._C_times_set: set[int] = set()
        self._D_times_set: set[int] = set()
        self._S_time_client_set: set[Tuple[int, int]] = set()
        self._P_time_client_set: set[Tuple[int, int]] = set()

    def get_e_alpha_gamma(self, t: int) -> torch.Tensor:
        Iz = torch.eye(self.d_z).double()
        bold_alpha = torch.block_diag(*[client.alpha * Iz for client in self.clients])
        bold_gamma = torch.block_diag(*[client.gamma * Iz for client in self.clients])
        e_alpha = torch.linalg.matrix_exp(-1 * bold_alpha * (self.sync_freq - t))
        # Output shape: Nd_z x Nd_z
        out = torch.matmul(e_alpha, bold_gamma)
        assert out.shape == (self.num_clients * self.d_z, self.num_clients * self.d_z)
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
            # P_t is Ndy x Nd_y
            self.set_client_pt(t=time, client_id=client_id, pt_value=(torch.matmul(W_current, W_current.T)))
            self.set_client_st(t=time, client_id=client_id, st_value=torch.matmul(-1 * W_current, latest_y.double()))

    def init_game_round_variables(self) -> None:

        self.init_loop_times()

        for client in self.clients:
            client.init_p_s(self.num_clients)
        self.e_alpha_gamma_A_inv = None

        # Go over past time steps and record these values for t in [0, T-1]
        self.A = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, self.num_clients * self.d_z),
            dtype=torch.float64,
        )
        self.B = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, self.num_clients * self.y_dim),
            dtype=torch.float64,
        )
        self.C = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, 1),
            dtype=torch.float64,
        )
        self.D = torch.zeros(
            (self.sync_freq, self.num_clients * self.y_dim, self.num_clients * self.d_z),
            dtype=torch.float64,
        )

    def calculate_a(self, time: int) -> torch.Tensor:
        At = torch.zeros(self.num_clients, self.num_clients, self.d_z, self.d_z, dtype=torch.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_A_ij_t(time, i, j)
                At[i][j] = item
        # A_t shape: N*N*d_z*d_z
        return At

    def calculate_b(self, t: int) -> torch.Tensor:
        B = []
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            B.append(torch.matmul(expected_e_ZT.transpose(0, 1).double(), client.P[t + 1].double()))
        return torch.stack(B).reshape(self.num_clients * self.d_z, self.num_clients)

    def calculate_c(self, t: int) -> torch.Tensor:
        C = []
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            C.append(torch.matmul(expected_e_ZT.transpose(0, 1).double(), client.S[t + 1].double()))
        return torch.cat(C, dim=1)

    def calculate_d(self, t: int) -> torch.Tensor:
        D = []
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            D.append(expected_e_ZT)
        # D's shape is Nd_y x Nd_z
        return torch.cat(D, dim=1)

    def set_A_t(self, t: int, A_t: torch.Tensor) -> None:
        self.A[t] = A_t.reshape(self.num_clients * self.d_z, self.num_clients * self.d_z)
        assert t not in self._A_times_set
        self._A_times_set.add(t)

    def get_A_t(self, t: int) -> torch.Tensor:
        assert t in self._A_times_set
        return self.A[t]

    def set_B_t(self, t: int, B_t: torch.Tensor) -> None:
        assert B_t.shape == (self.num_clients * self.d_z, self.num_clients * self.y_dim)
        self.B[t] = B_t
        assert t not in self._B_times_set
        self._B_times_set.add(t)

    def get_B_t(self, t: int) -> torch.Tensor:
        assert t in self._B_times_set
        return self.B[t]

    def set_C_t(self, t: int, C_t: torch.Tensor) -> None:
        assert C_t.shape == (self.num_clients * self.d_z, 1)
        self.C[t] = C_t
        assert t not in self._C_times_set
        self._C_times_set.add(t)

    def get_C_t(self, t: int) -> torch.Tensor:
        assert t in self._C_times_set
        return self.C[t]

    def set_D_t(self, t: int, D_t: torch.Tensor) -> None:
        assert D_t.shape == (self.num_clients * self.y_dim, self.num_clients * self.d_z)
        self.D[t] = D_t
        assert t not in self._D_times_set
        self._D_times_set.add(t)

    def get_D_t(self, t: int) -> torch.Tensor:
        assert t in self._D_times_set
        return self.D[t]

    def get_e_alpha_gamma_A_inv(self, t: int) -> torch.Tensor:
        A_t = self.get_A_t(t)
        e_alpha_gamma_t = self.get_e_alpha_gamma(t)  # output: Nd_z * Nd_z
        # output shape: N*dz*N*dz
        e_alpha_gamma_A = torch.add(e_alpha_gamma_t, A_t)
        assert e_alpha_gamma_A.shape == (self.num_clients * self.d_z, self.num_clients * self.d_z)
        return torch.inverse(e_alpha_gamma_A).double()

    def calculate_pt_client(
        self,
        t: int,
        client_id: int,
        e_alpha_gamma_A_inv: torch.Tensor,
        initial_term: torch.Tensor,
    ) -> torch.Tensor:
        client = self.clients[client_id]
        client_alpha = client.alpha
        client_gamma = client.gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        # e_i : shape [N*dy, dy]
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
        #  size should be (N*dz, N*dz)
        return torch.add(torch.add(term_1_2, term_3), term_4)

    def calculate_st_client(
        self, t: int, client_id: int, e_alpha_gamma_A_inv: torch.Tensor, wtyt: torch.Tensor
    ) -> torch.Tensor:
        client = self.clients[client_id]
        client_alpha = client.alpha
        client_gamma = client.gamma
        p_next = self.get_client_pt(t=(t + 1), client_id=client_id)
        s_next = self.get_client_st(t=(t + 1), client_id=client_id)
        e_client_alpha_t = torch.exp(torch.Tensor([(-1 * client_alpha) * (self.sync_freq - t)]))
        I_matrix = torch.eye(self.num_clients * self.d_z).double()

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
        # past_predictions is dy x N and B[t] is Nd_y x N
        # So we should change the shape of Y to d_yN x 1
        past_predictions = past_predictions.view(-1, 1)
        e_alpha_gamma_A_inv = self.get_e_alpha_gamma_A_inv(t)
        assert e_alpha_gamma_A_inv.shape == (self.num_clients * self.d_z, self.num_clients * self.d_z)
        bold_beta = torch.matmul(
            -1 * e_alpha_gamma_A_inv.double(),
            torch.add(
                torch.matmul(self.get_B_t(t).double(), past_predictions.double()),
                self.get_C_t(t).double(),
            ),
        )
        assert bold_beta.shape == (self.num_clients * self.d_z, 1)
        # bold_beta shape is Nd_z x 1
        # To be able to get beta for each client easier, we reshape bold beta
        bold_beta = bold_beta.reshape(-1, self.d_z, 1)
        return bold_beta


class TransformerGame(Game):

    def __init__(self, clients: List[Client], sync_freq: int, d_z: int) -> None:
        super().__init__(clients, sync_freq, d_z)

    def get_expectation_e_zt(self, t: int, client: Client) -> torch.Tensor:
        Z = client.feed_encoder(client.get_x(t).double())
        Z = Z.unsqueeze(1)
        e_i = client.get_e(self.num_clients)
        return torch.matmul(e_i.unsqueeze(1).double(), Z.transpose(0, 1).double())

    def get_A_ij_t(self, t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_E = self.get_expectation_e_zt(t, client_i)
        client_j_E = self.get_expectation_e_zt(t, client_j)
        return torch.matmul(torch.matmul(client_i_E.transpose(0, 1), client_i.P[t + 1]), client_j_E)


class EchoStateGame(Game):

    def __init__(
        self,
        clients: List[Client],
        sync_freq: int,
        d_z: int,
        N_samples: int = 100,
    ) -> None:
        super().__init__(clients, sync_freq, d_z)
        self.N_samples = N_samples

    def simulate_z_t(self, input: torch.Tensor, client: Client, Z_start: torch.Tensor) -> torch.Tensor:
        Z = Z_start
        #  Goes self.sync_freq back for simulation
        for _ in range(0, self.sync_freq):
            Z = client.encoder(input, Z, client.sigma)
        return Z

    def get_expectation_e_zt(self, t: int, client: Client, sampling: bool = True) -> torch.Tensor:
        input = client.get_x(t).double()
        assert type(client) is EchoStateNetworkClient
        if sampling:
            samples = []
            for i in range(self.N_samples):
                # Random init
                Z_start = client.get_latest_Z_T()
                Z = self.simulate_z_t(input, client, Z_start)
                Z = Z.unsqueeze(1)
                e_i = client.get_e(self.num_clients)
                e_z_T = torch.matmul(e_i.unsqueeze(1).double(), Z.transpose(0, 1).double())
                samples.append(e_z_T)
            sum_tensor = torch.zeros_like(samples[0])
            for sample in samples:
                sum_tensor += sample
            return sum_tensor / self.N_samples
        else:
            # One sample: one simulation
            Z_start = client.get_latest_Z_T()
            Z = self.simulate_z_t(input, client, Z_start)
            Z = Z.unsqueeze(1)
            e_i = client.get_e(self.num_clients)
            e_z_T = torch.matmul(e_i.unsqueeze(1).double(), Z.transpose(0, 1).double())
            return e_z_T

    def get_A_ij_t(self, t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        samples = []
        for i in range(self.N_samples):
            client_i_E = self.get_expectation_e_zt(t, client_i, False)
            client_j_E = self.get_expectation_e_zt(t, client_j, False)
            sample_value = torch.matmul(torch.matmul(client_i_E.transpose(0, 1), client_i.P[t + 1]), client_j_E)
            samples.append(sample_value)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor += sample
        return sum_tensor / self.N_samples


class RfnGame(Game):

    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        super().__init__(clients, sync_freq, z_dim)

    def get_a_t_embedding(self, time: int, client: Client) -> torch.Tensor:
        # some parts of the forwards pass (without randomness)
        a_t = (torch.matmul(client.encoder.A.double(), client.get_input_matrix(time).double())) + client.encoder.b
        assert a_t.shape == (self.y_dim, self.d_z)
        return a_t

    def get_expectation_zt(self, time: int, client: Client) -> torch.Tensor:
        a_t = self.get_a_t_embedding(time, client)
        phi = torch.distributions.Normal(0, 1).cdf((torch.mul((-1 / client.sigma), a_t)))
        one_bar = torch.ones(client.z_dim)
        exp_term = torch.exp((-1 / (2 * (client.sigma) ** 2)) * torch.mul(a_t, a_t))
        second_term = client.sigma / math.sqrt(2 * math.pi) * exp_term
        a_ti = torch.mul(a_t, (one_bar - phi)) + second_term
        assert a_ti.shape == (self.y_dim, self.d_z)
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
        return B_matrix

    def calculate_c(self, t: int) -> torch.Tensor:
        C = []
        #  client.S[t + 1] shape: Nd_y * 1
        for client in self.clients:
            a_ti = self.get_expectation_zt(t, client)  # shape: 1*d_z
            e_i = client.get_e(self.num_clients)  # shape: Nd_y*d_y
            C.append(
                torch.matmul(
                    torch.matmul(self.get_client_st(t=(t + 1), client_id=client.id).T.double(), e_i.double()), a_ti
                )
            )
        C_matrix = torch.cat(C, dim=1).T
        # C_matrix's shape is Nd_z x 1
        return C_matrix

    def calculate_d(self, t: int) -> torch.Tensor:
        D = []
        for client in self.clients:
            a_ti = self.get_expectation_zt(t, client)  # shape: 1*d_z
            e_i = client.get_e(self.num_clients)  # shape: Nd_y*d_y
            D.append(torch.matmul(e_i, a_ti))
        # D's shape is Nd_y x Nd_z
        return torch.cat(D, dim=1)

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
            assert A_ij.shape == (self.d_z, self.d_z)
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
            assert A_ii.shape == (self.d_z, self.d_z)
            return A_ii
