import math
from typing import List

import torch

from fedmoe.clients.client import Client
from fedmoe.game.game import Game


class RfnGame(Game):
    """
    The specific case of Random Feature Network game where dy = 1.
    """

    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        for client in clients:
            if client.y_dim != 1:
                print("WARNING: The game is currently only well defined for dy=1")
        self.standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        super().__init__(clients, sync_freq, z_dim)

    def get_a_t_embedding(self, time: int, client: Client) -> torch.Tensor:
        # some parts of the forwards pass (without randomness)
        a_t = (
            torch.matmul(client.encoder.A.double(), self.get_input_matrix(time, client).double())
        ) + client.encoder.b.double()
        assert a_t.shape == (self.y_dim, self.z_dim)
        return a_t

    def standard_normal_cdf(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.standard_normal.cdf(input_tensor)

    def get_expectation_zt(self, time: int, client: Client) -> torch.Tensor:
        a_t = self.get_a_t_embedding(time, client)
        phi = self.standard_normal_cdf((torch.mul((-1 / client.sigma), a_t)))
        one_bar = torch.ones(client.z_dim)
        exp_term = torch.exp((-1 / (2 * (client.sigma) ** 2)) * torch.mul(a_t, a_t))
        second_term = client.sigma / math.sqrt(2 * math.pi) * exp_term
        # a(t,i) is the expectation of Z_{t+1}^i in lemma 2 appendix, but we use it as Z_t^i in the game
        a_z_ti = torch.mul(a_t, (one_bar - phi)) + second_term
        assert a_z_ti.shape == (self.y_dim, self.z_dim)
        return a_z_ti

    def get_expectation_e_zt(self, game_t: int, client: Client) -> torch.Tensor:
        """
        Computes "$e_i a(t, i)$" for each client i
        """
        # We don't need to feed the transformer again.
        a_z_ti = self.get_expectation_zt(game_t, client).double()
        # Embedding shape is y_dim x z_dim
        assert a_z_ti.shape == (self.y_dim, self.z_dim)
        # e_i's shape is (num_clients * self.y_dim, self.y_dim)
        e_i = client.get_e(self.num_clients)
        # output shape is Ny_dim x z_dim
        return torch.matmul(
            e_i.double(),
            a_z_ti.double(),
        )

    def compute_ii_sub_matrices(self, time: int, client_i: Client, first_term: torch.Tensor) -> torch.Tensor:
        """
        Computes the sub-matrices of A_ii, A_hat_ii and D_i^{jj} for the given time and client.
        This function iterates over a matrix of size d_z by d_z and fills its elements.

        Args:
            time (int): game time.
            client_i (Client): client referring to index i.
            first_term (torch.Tensor): It is the value multiplied with the rest of the elements in the sub-matrices.
                Depending on the matrix, it can be P_{t+1}^{ii} used for A and D_i, or w_t^{(i)} **2 used for A_hat.
        """
        matrix_ii = torch.zeros(self.z_dim, self.z_dim, dtype=torch.float64)
        a_z_ti = self.get_expectation_zt(time, client_i)
        at = self.get_a_t_embedding(time, client_i)
        # Instead of j, k from 0 to d_z in the equations, we use p, k to distinguish from the client indices.
        for p in range(self.z_dim):
            # p is j in equations.
            for k in range(self.z_dim):
                if p == k:
                    phi = self.standard_normal_cdf(-at[:, p] / client_i.sigma)
                    second_term = (torch.mul(at, at)[:, p] + client_i.sigma**2) * (1 - phi) + at[:, p] * (
                        client_i.sigma / math.sqrt(2 * math.pi)
                    ) * torch.exp(-1 * (torch.mul(at, at)[:, p]) / (2 * (client_i.sigma**2)))
                else:
                    second_term = a_z_ti[:, p] * a_z_ti[:, k]
                matrix_ii[p][k] = first_term * second_term
        assert matrix_ii.shape == (self.z_dim, self.z_dim)
        return matrix_ii

    def get_A_ij_t(self, time: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        # Check that client_i.P[time + 1] is set
        next_p_i = self.get_client_pt(t=(time + 1), client_id=i).reshape(
            (self.num_clients, self.num_clients, self.y_dim, self.y_dim)
        )
        # a_z_ti is a(t,i) which is the E[Z_t^i] for RFNs; shape: d_y*d_z
        # at is a partial forward pass without randomness used in a(t,i) calculations. shape: d_y*d_z
        if i != j:
            a_z_ti = self.get_expectation_zt(time, client_i)
            a_z_tj = self.get_expectation_zt(time, client_j)
            A_ij = torch.matmul(((a_z_ti.T * next_p_i[i][j])), a_z_tj)
        else:
            # i == j
            A_ij = self.compute_ii_sub_matrices(time, client_i, next_p_i[i][j])
        assert A_ij.shape == (self.z_dim, self.z_dim)
        return A_ij

    def get_A_hat_ij_t(self, time: int, i: int, j: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        # Since dy = 1, the bold w_t is basically the same as w_t (mixture weights)
        assert bold_w_t.shape == (self.num_clients, 1)
        client_i = self.clients[i]
        client_j = self.clients[j]
        a_z_ti = self.get_expectation_zt(time, client_i)
        a_z_tj = self.get_expectation_zt(time, client_j)
        if i != j:
            A_hat_ij = torch.matmul(((a_z_ti.T * bold_w_t[i]) * bold_w_t[j]), a_z_tj)
        else:
            # i == j
            first_term = bold_w_t[i] * bold_w_t[j]
            A_hat_ij = self.compute_ii_sub_matrices(time, client_i, first_term)

        # A_hat_ij shape: d_z*d_z
        assert A_hat_ij.shape == (self.z_dim, self.z_dim)
        return A_hat_ij

    def get_D_client_ij_t(self, time: int, i: int, j: int, P_t_plus_1_client: torch.Tensor) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        assert P_t_plus_1_client.shape == (self.num_clients, self.num_clients)
        a_z_ti = self.get_expectation_zt(time, client_i)
        a_z_tj = self.get_expectation_zt(time, client_j)
        if i != j:
            D_client_ij = torch.matmul((a_z_ti.T * P_t_plus_1_client[i][j]), a_z_tj)
        else:
            # i == j
            D_client_ij = self.compute_ii_sub_matrices(time, client_i, P_t_plus_1_client[i][j])
        assert D_client_ij.shape == (self.z_dim, self.z_dim)
        return D_client_ij
