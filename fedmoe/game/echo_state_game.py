from typing import List

import torch

from fedmoe.clients.client import Client
from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.game.game import Game

torch.set_default_dtype(torch.float64)


class EchoStateGame(Game):
    def __init__(
        self,
        clients: List[Client],
        sync_freq: int,
        z_dim: int,
        n_samples: int = 100,
    ) -> None:
        super().__init__(clients, sync_freq, z_dim)
        self.n_samples = n_samples

    def simulate_z_t(self, game_t: int, client: Client) -> torch.Tensor:
        # Setting z start, which is the last z used in the last sync step.
        # Based on the game time scale, it is game_t=0.
        Z = self.get_z(game_t=0, client=client)
        # Starts the simulation from 1 (last sync step + 1) to desired current t.
        # First Z used in the simulation is Z_0, so the first x used should be x_1.
        for back_game_t in range(1, game_t + 1):
            # Z_t (back_game_t) is generated.
            Z = client.encoder(self.get_input_matrix(back_game_t, client), Z, client.sigma)
        # The final Z estimated is Z_{game_t} or the current Z in game.
        assert Z.shape == (self.y_dim, self.z_dim)
        return Z

    def get_expectation_e_zt(self, game_t: int, client: Client) -> torch.Tensor:
        # Note that game_t here is not server time, but rather game time [0, sync_freq]
        assert type(client) is EchoStateNetworkClient
        samples = []
        for _ in range(self.n_samples):
            # Start from time current_t-T for the trajectory
            # If we're PREDICTING t=9 and we are at t=8, with sync frequency T=4, then we want to generate trajectories
            # Z_4 -> Z_5 -> Z_6 -> Z_7 for our Nash game.
            # This means we start from Z_4 and use x_5, x_6, x_7 to generate these latent values.
            # To get Z_6, we again start from Z_4 -> Z_5 -> Z_6
            estimated_z_t = self.simulate_z_t(game_t, client)
            samples.append(estimated_z_t)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor = sum_tensor + sample
        average_estimated_z_t = sum_tensor / self.n_samples
        e_i = client.get_e(self.num_clients)
        estimated_e_z_t = torch.matmul(e_i.double(), average_estimated_z_t.double())
        return estimated_e_z_t

    def get_expectation_e_z_transpose_P_e_z(self, game_t: int, client: Client, next_p_i: torch.Tensor) -> torch.Tensor:
        # Note that game_t here is not server time, but rather game time [0, sync_freq]
        # When the z_t values are the same in these expectation calculations, we can't compute the expectations
        # separately. So we do everything together.
        assert type(client) is EchoStateNetworkClient
        samples = []
        for _ in range(self.n_samples):
            # Start from time current_t-T for the trajectory
            # If we're PREDICTING t=9 and we are at t=8, with sync frequency T=4, then we want to generate trajectories
            # Z_4 -> Z_5 -> Z_6 -> Z_7 -> Z_8 for our Nash game.
            # This means we start from Z_4 and use x_5, x_6, x_7 to generate these latent values.
            Z_i = self.simulate_z_t(game_t, client)
            e_i = client.get_e(self.num_clients)
            estimated_e_z_i = torch.matmul(e_i.double(), Z_i.double())
            sample = torch.matmul(torch.matmul(estimated_e_z_i.T, next_p_i), estimated_e_z_i)
            samples.append(sample)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor = sum_tensor + sample
        return sum_tensor / self.n_samples

    def get_A_ij_t(self, game_t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        next_p_i = self.get_client_pt(t=(game_t + 1), client_id=i)
        if i != j:
            # If we're in the off diagonal entries of A_{ij}(t) then the expectations are independent
            # So we can calculate everything separately
            client_i_e_z = self.get_expectation_e_zt(game_t, client_i)
            client_j_e_z = self.get_expectation_e_zt(game_t, client_j)
            return torch.matmul(torch.matmul(client_i_e_z.T, next_p_i), client_j_e_z)
        else:
            # If we're on the diagonal, then the Z_ts are not independent and we have to do an estimation of everything
            # combined
            return self.get_expectation_e_z_transpose_P_e_z(game_t, client_i, next_p_i)

    def get_expectation_e_z_transpose_wwT_e_z(
        self, game_t: int, client: Client, bold_w_t: torch.Tensor
    ) -> torch.Tensor:
        assert type(client) is EchoStateNetworkClient
        samples = []
        for _ in range(self.n_samples):
            # Start from time current_t-T for the trajectory
            # If we're PREDICTING t=9 and we are at t=8, with sync frequency T=4, then we want to generate trajectories
            # Z_4 -> Z_5 -> Z_6 -> Z_7 -> Z_8 for our Nash game.
            # This means we start from Z_4 and use x_5, x_6, x_7 to generate these latent values.
            Z_i = self.simulate_z_t(game_t, client)
            e_i = client.get_e(self.num_clients)
            estimated_e_z_i = torch.matmul(e_i.double(), Z_i.double())
            sample = torch.matmul(
                torch.matmul(torch.matmul(estimated_e_z_i.T, bold_w_t), bold_w_t.T), estimated_e_z_i
            ).double()
            samples.append(sample)
        sum_tensor = torch.zeros_like(samples[0])
        for sample in samples:
            sum_tensor = sum_tensor + sample
        return sum_tensor / self.n_samples

    def get_A_hat_ij_t(self, time: int, i: int, j: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        assert bold_w_t.shape == (self.num_clients * self.y_dim, self.y_dim)
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_e_z_t = self.get_expectation_e_zt(time, client_i)
        client_j_e_z_t = self.get_expectation_e_zt(time, client_j)
        if i != j:
            A_hat_ij = torch.matmul(torch.matmul(torch.matmul(client_i_e_z_t.T, bold_w_t), bold_w_t.T), client_j_e_z_t)
        else:
            # i == j
            A_hat_ij = self.get_expectation_e_z_transpose_wwT_e_z(time, client_i, bold_w_t)

        # A_hat_ij shape: d_z*d_z
        assert A_hat_ij.shape == (self.z_dim, self.z_dim)
        return A_hat_ij

    def get_D_client_ij_t(self, time: int, i: int, j: int, P_t_plus_1_client: torch.Tensor) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        assert P_t_plus_1_client.shape == (self.num_clients * self.y_dim, self.num_clients * self.y_dim)
        client_i_e_z_t = self.get_expectation_e_zt(time, client_i)
        client_j_e_z_t = self.get_expectation_e_zt(time, client_j)
        if i != j:
            D_client_ij = torch.matmul(torch.matmul(client_i_e_z_t.T, P_t_plus_1_client), client_j_e_z_t)
        else:
            # i == j
            D_client_ij = self.get_expectation_e_z_transpose_P_e_z(time, client_i, P_t_plus_1_client)
        assert D_client_ij.shape == (self.z_dim, self.z_dim)
        return D_client_ij
