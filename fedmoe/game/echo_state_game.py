from typing import List

import torch

from fedmoe.clients.client import Client
from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.game.game import Game


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
