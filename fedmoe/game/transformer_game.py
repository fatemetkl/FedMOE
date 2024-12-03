from typing import List

import torch

from fedmoe.clients.client import Client
from fedmoe.game.game import Game


class TransformerGame(Game):
    def __init__(self, clients: List[Client], sync_freq: int, z_dim: int) -> None:
        super().__init__(clients, sync_freq, z_dim)

    # def get_input(self, game_t: int, client: Client) -> torch.Tensor:
    #     """
    #     Maps the time game_t in the game (between 0 to sync_freq) to the time scale used in the server,
    #     current_time, and returns the input (x_t) associated with server time.
    #     """
    #     server_time = self.map_game_time_to_server_time(game_t, client)
    #     # Assuming that the input shape in transformer is (x_dim, 1)
    #     return client.get_x(server_time), server_time

    def get_hidden_state(self, game_t: int, client: Client) -> torch.Tensor:
        """
        Maps the time game_t in the game (between 0 to sync_freq) to the time scale used in the server,
        current_time, and returns the hidden state (z_t) associated with server time.
        """
        server_time = self.map_game_time_to_server_time(game_t, client)
        # Assuming that the hidden state shape in transformer is (z_dim, 1)
        return client.state.get_hidden_state_t(server_time)

    def get_expectation_e_zt(self, game_t: int, client: Client) -> torch.Tensor:
        """
        Computes "$e_i phi^{(i)}(x_t)$" for each client i
        """
        # We don't need to feed the transformer again.
        Z = self.get_hidden_state(game_t, client).double()
        # Embedding shape is y_dim x z_dim
        assert Z.shape == (self.y_dim, self.z_dim)
        # e_i's shape is (num_clients * self.y_dim, self.y_dim)
        e_i = client.get_e(self.num_clients)
        # output shape is Ny_dim x z_dim
        return torch.matmul(
            e_i.double(),
            Z.double(),
        )

    def get_A_ij_t(self, game_t: int, i: int, j: int) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_E = self.get_expectation_e_zt(game_t, client_i)
        client_j_E = self.get_expectation_e_zt(game_t, client_j)
        return torch.matmul(torch.matmul(client_i_E.T, client_i.P[game_t + 1]), client_j_E)

    def get_A_hat_ij_t(self, game_t: int, i: int, j: int, bold_w_t: torch.Tensor) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_E = self.get_expectation_e_zt(game_t, client_i)
        client_j_E = self.get_expectation_e_zt(game_t, client_j)
        return torch.matmul(torch.matmul(client_i_E.T, torch.matmul(bold_w_t, bold_w_t.T)), client_j_E)

    def get_D_client_ij_t(self, game_t: int, i: int, j: int, P_t_plus_1_client: torch.Tensor) -> torch.Tensor:
        client_i = self.clients[i]
        client_j = self.clients[j]
        client_i_E = self.get_expectation_e_zt(game_t, client_i)
        client_j_E = self.get_expectation_e_zt(game_t, client_j)
        return torch.matmul(torch.matmul(client_i_E.T, P_t_plus_1_client), client_j_E)
