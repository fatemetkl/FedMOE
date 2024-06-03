import math
from abc import ABC, abstractmethod
from typing import List

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
        self.e_alpha_gamma = self.init_e_alpha_gamma()
        self.y_dim = self.clients[0].y_dim

    @abstractmethod
    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        raise NotImplementedError

    def init_e_alpha_gamma(self) -> torch.Tensor:
        bold_alpha = torch.diag(torch.tensor([-1 * client.alpha for client in self.clients]))
        bold_gamma = torch.diag(torch.tensor([client.gamma for client in self.clients]))
        e_alpha = torch.linalg.matrix_exp(bold_alpha)
        return torch.matmul(e_alpha, bold_gamma)

    def init_game_round_variables(self, latest_mixture_weights: torch.Tensor, y_T: torch.Tensor) -> None:
        for client in self.clients:
            client.init_p_s(self.num_clients)
        self.e_alpha_gamma_A_inv = None

        self.W_current = torch.transpose(
            torch.tensor(
                [torch.matmul(w_Tn.double(), torch.eye(self.y_dim).double()) for w_Tn in latest_mixture_weights]
            ).reshape(1, -1),
            0,
            1,
        )

        for client in self.clients:
            client.P[self.sync_freq] = torch.matmul(self.W_current, self.W_current.transpose(0, 1))
            client.S[self.sync_freq] = torch.matmul(-1 * self.W_current, y_T.double())
        # Go over past time steps and record these values for t in [0, T-1]
        self.A = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, self.num_clients * self.d_z),
            dtype=torch.float64,
        )
        self.B = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, self.num_clients),
            dtype=torch.float64,
        )
        self.C = torch.zeros(
            (self.sync_freq, self.num_clients * self.d_z, self.y_dim),
            dtype=torch.float64,
        )
        self.D = torch.zeros(
            (self.sync_freq, self.num_clients, self.num_clients * self.d_z),
            dtype=torch.float64,
        )

    def calculate_a(self, time: int) -> torch.Tensor:
        At = torch.zeros(self.num_clients, self.num_clients, self.d_z, self.d_z, dtype=torch.float64)
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                item = self.get_A_ij_t(time, i, j)
                At[i][j] = item
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
        return torch.stack(C).reshape(self.num_clients * self.d_z, self.y_dim)

    def calculate_d(self, t: int) -> torch.Tensor:
        D = []
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            D.append(expected_e_ZT)
        return torch.stack(D).reshape(self.num_clients, self.num_clients * self.d_z)

    def set_A_t(self, t: int, A_t: torch.Tensor) -> None:
        self.A[t] = A_t.reshape(self.num_clients * self.d_z, self.num_clients * self.d_z)

    def set_B_t(self, t: int, B_t: torch.Tensor) -> None:
        self.B[t] = B_t

    def set_C_t(self, t: int, C_t: torch.Tensor) -> None:
        self.C[t] = C_t

    def set_D_t(self, t: int, D_t: torch.Tensor) -> None:
        self.D[t] = D_t

    def get_e_alpha_gamma_A_inv(self, t: int) -> torch.Tensor:
        A_t = self.A[t].reshape(self.num_clients, self.num_clients, self.d_z, self.d_z)
        temp = torch.zeros(
            self.num_clients,
            self.num_clients,
            self.d_z,
            self.d_z,
            dtype=torch.float64,
        )
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                # sub_matrix: shape dz*dz
                temp[i][j] = torch.add(A_t[i][j], self.e_alpha_gamma[i][j])
        # output shape: N*N*dz*dz
        temp = temp.reshape(self.num_clients * self.d_z, self.num_clients * self.d_z)
        e_alpha_gamma_A_inv = torch.inverse(temp)
        e_alpha_gamma_A_inv = e_alpha_gamma_A_inv.reshape((self.num_clients * self.d_z, self.num_clients * self.d_z))
        return e_alpha_gamma_A_inv

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
        p_next = client.P[t + 1]
        # e_i : shape [dz, N*dz]
        bold_e_i = torch.kron(client.get_e(self.num_clients).unsqueeze(1), torch.eye(self.d_z))
        e_neg_alpha = torch.exp(torch.tensor(-1 * client_alpha))

        I_matrix = torch.eye(self.num_clients).double()

        x = torch.matmul(self.D[t].double(), e_alpha_gamma_A_inv)
        m = torch.matmul(x.double(), self.B[t].double())
        term_1_pre = (m - I_matrix).transpose(0, 1)

        # New: client_alpha*term_1_pre

        # term_1 shape: N*N
        term_1 = torch.matmul(e_neg_alpha * term_1_pre, p_next)
        term_2 = torch.matmul(torch.matmul(self.D[t], e_alpha_gamma_A_inv).double(), self.B[t].double()) - I_matrix
        # term_2 shape: N*N
        term_1_2 = torch.matmul(term_1, term_2)
        # dz*N*dz
        # D shape: N, N*dz
        # Previously ->
        # bold_e_i = bold_e_i.reshape(num_clients, d_z * d_z)
        # bold_e_i_T = torch.matmul(bold_e_i.transpose(0, 1), D_t)
        # Now ->
        bold_e_i_T = bold_e_i.transpose(0, 1)

        term_3 = torch.matmul(torch.matmul(bold_e_i_T, e_alpha_gamma_A_inv).double(), self.B[t].double()).transpose(
            0, 1
        )
        term_4 = torch.matmul(torch.matmul(bold_e_i_T.double(), e_alpha_gamma_A_inv), self.B[t])
        term_3_4 = torch.matmul(client_gamma * term_3, term_4)
        p_term = torch.add(term_1_2, term_3_4) + initial_term

        return p_term

    def calculate_st_client(
        self, t: int, client_id: int, e_alpha_gamma_A_inv: torch.Tensor, wtyt: torch.Tensor
    ) -> torch.Tensor:
        client = self.clients[client_id]
        client_alpha = client.alpha
        client_gamma = client.gamma
        p_next = client.P[t + 1]
        s_next = client.S[t + 1]
        # e_i : shape [dz, N*dz]
        bold_e_i = torch.kron(client.get_e(self.num_clients).unsqueeze(1), torch.eye(self.d_z))

        e_pow_neg_alpha = torch.exp(torch.tensor(-1 * client_alpha))

        term_1_1 = torch.matmul(self.B[t].transpose(0, 1), e_alpha_gamma_A_inv)
        DPD = torch.matmul(torch.matmul(e_pow_neg_alpha * self.D[t].transpose(0, 1), p_next), self.D[t])
        term_1_2 = torch.add(torch.matmul(client_gamma * bold_e_i, bold_e_i.transpose(0, 1)), DPD)
        term_1_3 = torch.matmul(e_alpha_gamma_A_inv, self.C[t])
        term_2 = torch.matmul(
            torch.matmul(torch.matmul(e_pow_neg_alpha * p_next, self.D[t]), e_alpha_gamma_A_inv),
            self.C[t],
        )
        term_3 = torch.matmul(
            torch.matmul(
                torch.matmul(e_pow_neg_alpha * self.B[t].transpose(0, 1), e_alpha_gamma_A_inv),
                self.D[t].transpose(0, 1),
            ),
            s_next,
        )
        s_term = torch.matmul(torch.matmul(term_1_1, term_1_2), term_1_3) - term_2 - term_3
        s_term = s_term + e_pow_neg_alpha * s_next - wtyt

        return s_term

    def set_client_pt(self, t: int, client_id: int, pt_value: torch.Tensor) -> None:
        self.clients[client_id].P[t] = pt_value

    def set_client_st(self, t: int, client_id: int, st_value: torch.Tensor) -> None:
        self.clients[client_id].S[t] = st_value

    def compute_beta(self, t: int, past_predictions: List[torch.Tensor]) -> torch.Tensor:
        e_alpha_gamma_A_inv = self.get_e_alpha_gamma_A_inv(t)
        return torch.matmul(
            -1 * e_alpha_gamma_A_inv.double(),
            torch.add(
                torch.matmul(self.B[t].double(), past_predictions[t].transpose(0, 1).double()),
                self.C[t].double(),
            ),
        )


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

    def __init__(self, clients: List[Client], sync_freq: int, d_z: int) -> None:
        super().__init__(clients, sync_freq, d_z)

    def get_a_t(self, time: int, client: Client) -> torch.Tensor:
        # some part of the forwards pass (without randomness)
        a_t = (torch.matmul(client.encoder.A.float(), client.get_x(time).float()).squeeze(1)) + client.encoder.b
        return a_t

    def get_expectation_zt(self, time: int, client: Client) -> torch.Tensor:
        a_t = self.get_a_t(time, client)
        phi = torch.distributions.Normal(0, 1).cdf((torch.mul((-1 / client.sigma), a_t)))
        one_bar = torch.ones(client.d_z)
        exp_term = torch.exp(-1 * torch.square(a_t) / (2 * client.sigma))
        second_term = math.sqrt(client.sigma / 2 * math.pi) * exp_term

        a_ti = torch.mul(a_t, (one_bar - phi)) + second_term

        return a_ti

    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        a_ti = self.get_expectation_zt(time, client)
        a_ti = a_ti.unsqueeze(1)
        e_i = client.get_e(self.num_clients)
        return torch.matmul(e_i.unsqueeze(1).double(), a_ti.transpose(0, 1).double())

    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        if i != j:
            a_ti = self.get_expectation_zt(time, client_i)
            a_tj = self.get_expectation_zt(time, client_j)
            a_tj = a_tj.unsqueeze(1)
            A_ij = torch.matmul((a_ti * client_i.P[time + 1][i][j]).unsqueeze(1), a_tj.transpose(0, 1))

            return A_ij
        else:
            A_ii = torch.zeros(client_i.d_z, client_i.d_z, dtype=torch.float64)
            a_ti = self.get_expectation_zt(time, client_i)
            at = self.get_a_t(time, client_i)

            for p in range(client_i.d_z):
                for k in range(client_i.d_z):
                    if p == k:
                        phi = torch.distributions.Normal(0, 1).cdf(-at[p] / client_i.sigma)

                        second_term = torch.mul((at[p] ** 2 + client_i.sigma), (1 - phi)) + at[p] * math.sqrt(
                            client_i.sigma / 2 * math.pi
                        ) * torch.exp(-1 * (at[p] ** 2) / (2 * client_i.sigma))
                    else:
                        second_term = a_ti[p] * a_ti[k]

                    A_ii[p][k] = client_i.P[time + 1][i][i] * second_term
            return A_ii
