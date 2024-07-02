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
        self.y_dim = self.clients[0].y_dim

    @abstractmethod
    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        raise NotImplementedError

    def get_e_alpha_gamma(self, t: int) -> torch.Tensor:
        Iz = torch.eye(self.d_z).double()
        bold_alpha = torch.block_diag(*[client.alpha * Iz for client in self.clients])
        bold_gamma = torch.block_diag(*[client.gamma * Iz for client in self.clients])
        e_alpha = torch.linalg.matrix_exp(-1 * bold_alpha * (self.sync_freq - t))
        # Output shape: Nd_z * Nd_z
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
        A_t = self.A[t]
        e_alpha_gamma_t = self.get_e_alpha_gamma(t)  # output: Nd_z * Nd_z
        # output shape: N*dz*N*dz
        e_alpha_gamma_A = torch.add(e_alpha_gamma_t, A_t)
        return torch.inverse(e_alpha_gamma_A)

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
        # e_i : shape [N*dy, dy]
        e_client_alpha_t = torch.exp(torch.tensor(-1 * client_alpha) * (self.sync_freq - t))
        e_client_alpha_t_gamma = e_client_alpha_t * client_gamma

        I_matrix = torch.eye(self.num_clients * self.y_dim).double()

        x = torch.matmul(self.D[t].double(), e_alpha_gamma_A_inv)
        m = torch.matmul(x.double(), self.B[t].double())
        term_1_pre = (m - I_matrix).transpose(0, 1)

        # term_1 shape: N*N
        term_1 = torch.matmul(term_1_pre, p_next)
        term_2 = torch.matmul(torch.matmul(self.D[t], e_alpha_gamma_A_inv).double(), self.B[t].double()) - I_matrix
        # term_2 shape: N*N
        term_1_2 = torch.matmul(term_1, term_2)

        term_3_part2 = torch.matmul(e_alpha_gamma_A_inv, self.B[t].double())
        term_3_part1 = e_client_alpha_t_gamma * term_3_part2.transpose(0, 1)
        term_3 = torch.matmul(term_3_part1, term_3_part2)
        term_4 = e_client_alpha_t * initial_term
        #  size should be (N*dz, N*dz)
        return term_1_2 + term_3 + term_4

    def calculate_st_client(
        self, t: int, client_id: int, e_alpha_gamma_A_inv: torch.Tensor, wtyt: torch.Tensor
    ) -> torch.Tensor:
        client = self.clients[client_id]
        client_alpha = client.alpha
        client_gamma = client.gamma
        p_next = client.P[t + 1]
        s_next = client.S[t + 1]
        e_client_alpha_t = torch.exp(torch.tensor(-1 * client_alpha) * (self.sync_freq - t))
        I_matrix = torch.eye(self.num_clients * self.d_z).double()

        term_1_1 = torch.matmul(self.B[t].transpose(0, 1), e_alpha_gamma_A_inv)
        DPD = torch.matmul(torch.matmul(self.D[t].transpose(0, 1), p_next), self.D[t])
        term_1_2 = torch.add(e_client_alpha_t * client_gamma * I_matrix, DPD)
        term_1_3 = torch.matmul(e_alpha_gamma_A_inv, self.C[t])
        term_2 = torch.matmul(
            torch.matmul(torch.matmul(p_next, self.D[t]), e_alpha_gamma_A_inv),
            self.C[t],
        )
        term_3 = torch.matmul(
            torch.matmul(
                torch.matmul(self.B[t].transpose(0, 1), e_alpha_gamma_A_inv),
                self.D[t].transpose(0, 1),
            ),
            s_next,
        )
        s_term = torch.matmul(torch.matmul(term_1_1, term_1_2), term_1_3) - term_2 - term_3
        s_term = s_term + s_next - e_client_alpha_t * wtyt

        return s_term

    def set_client_pt(self, t: int, client_id: int, pt_value: torch.Tensor) -> None:
        self.clients[client_id].P[t] = pt_value

    def set_client_st(self, t: int, client_id: int, st_value: torch.Tensor) -> None:
        self.clients[client_id].S[t] = st_value

    def compute_beta(self, t: int, past_predictions: torch.Tensor) -> torch.Tensor:
        e_alpha_gamma_A_inv = self.get_e_alpha_gamma_A_inv(t)
        return torch.matmul(
            -1 * e_alpha_gamma_A_inv.double(),
            torch.add(
                torch.matmul(self.B[t].double(), past_predictions.transpose(0, 1).double()),
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
        # some parts of the forwards pass (without randomness)
        a_t = (torch.matmul(client.encoder.A.float(), client.get_x(time).float()).squeeze(1)) + client.encoder.b
        return a_t

    def get_expectation_zt(self, time: int, client: Client) -> torch.Tensor:
        a_t = self.get_a_t(time, client)
        phi = torch.distributions.Normal(0, 1).cdf((torch.mul((-1 / client.sigma), a_t)))
        one_bar = torch.ones(client.d_z)
        exp_term = torch.exp((-1 / (2 * (client.sigma) ** 2)) * torch.mul(a_t, a_t))
        second_term = client.sigma / math.sqrt(2 * math.pi) * exp_term
        a_ti = torch.mul(a_t, (one_bar - phi)) + second_term

        return a_ti

    def get_expectation_e_zt(self, time: int, client: Client) -> torch.Tensor:
        a_ti = self.get_expectation_zt(time, client)  # shape: 1*d_z
        e_i = client.get_e(self.num_clients)  # shape: N*1
        return torch.matmul(e_i.double(), a_ti.double())  # output shape: Nd_y*d_z

    def calculate_b(self, t: int) -> torch.Tensor:
        B = []
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            B.append(torch.matmul(client.P[t + 1].double(), expected_e_ZT.double()))
        return torch.stack(B).reshape(self.num_clients * self.d_z, self.num_clients)

    def calculate_c(self, t: int) -> torch.Tensor:
        C = []
        #  client.S[t + 1] shape: Nd_y * 1
        for client in self.clients:
            expected_e_ZT = self.get_expectation_e_zt(t, client)
            C.append(torch.matmul(client.S[t + 1].transpose(0, 1).double(), expected_e_ZT.double()))
        assert self.y_dim == 1
        # output shape: Nd_z * 1
        return torch.stack(C).reshape(self.num_clients * self.d_z, 1)

    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        next_p = client_i.P[time + 1].reshape((self.num_clients, self.num_clients, self.y_dim, self.y_dim))
        # a_ti is a(t,i) which is the E[Z_t^i] for RFNs; shape: 1*d_z
        # at is a partial forward pass without randomness used in a(t,i) calculations. shape: 1*d_Z
        if i != j:
            a_ti = self.get_expectation_zt(time, client_i)
            a_tj = self.get_expectation_zt(time, client_j)
            A_ij = torch.matmul((a_ti.transpose(0, 1) * next_p[i][j]), a_tj)
            # A_ij shape: d_z*d_z
            return A_ij
        else:
            A_ii = torch.zeros(client_i.d_z, client_i.d_z, dtype=torch.float64)
            #  Shape of the following to tensors: 1*d_Z --> squeeze to remove extra first dimension
            a_ti = self.get_expectation_zt(time, client_i).squeeze(0)
            at = self.get_a_t(time, client_i).squeeze(0)

            for p in range(client_i.d_z):
                for k in range(client_i.d_z):
                    if p == k:
                        phi = torch.distributions.Normal(0, 1).cdf(-at[p] / client_i.sigma)

                        second_term = (torch.mul(at, at)[p] + client_i.sigma**2) * (1 - phi) + at[
                            p
                        ] * client_i.sigma / math.sqrt(2 * math.pi) * torch.exp(
                            -1 * (torch.mul(at, at)[p]) / (2 * (client_i.sigma**2))
                        )
                    else:
                        second_term = a_ti[p] * a_ti[k]
                    # Considering that dy=1
                    assert self.y_dim == 1
                    A_ii[p][k] = next_p[i][i] * second_term
            # A_ii shape: d_z*d_z
            return A_ii
