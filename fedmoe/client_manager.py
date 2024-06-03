from typing import List

import torch

from fedmoe.clients.client import Client, ClientType
from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.clients.rfn_client import RandomFeatureNetworkClient
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.datasets.echotorch_datasets.periodic_signal import PeriodicSignalDataset


class ClientManager:
    def __init__(
        self, client_type: ClientType, num_clients: int, sync_freq: int, d_z: int, alpha: float, gamma: float
    ) -> None:
        self.client_type = client_type
        self.num_clients = num_clients
        self.d_z = d_z
        self.alpha = alpha
        self.gamma = gamma
        self.data_length = 200
        self.n_samples = 40
        self.y_dim = 1
        self.data = self.set_data(self.n_samples)
        self.common_target_sequence = self.set_target()
        self.clients = self.initiate_clients(num_clients, sync_freq)

    def set_data(self, n_samples: int) -> torch.Tensor:
        #  For now we assume there is only one data sequence
        periodic_ds = PeriodicSignalDataset(self.data_length, n_samples=n_samples, period=[i for i in range(9)])
        return torch.cat([periodic_ds[i] for i in range(0, len(periodic_ds))]).reshape(-1)

    def set_target(self) -> torch.Tensor:
        assert self.data is not None, " data should be set first"
        # Target is the shifted input to the left
        return self.data[1:]

    def initiate_clients(self, num_clients: int, sync_freq: int) -> List[Client]:
        clients: List[Client] = []
        #  Every tensor type is float64
        for i in range(num_clients):
            client: RandomFeatureNetworkClient | TransformerClient | EchoStateNetworkClient
            if self.client_type == ClientType.RFN:
                client = RandomFeatureNetworkClient(
                    id=i,
                    sync_steps=sync_freq,
                    d_z=self.d_z,
                    data_length=self.data_length,
                    y_dim=self.y_dim,
                )
            elif self.client_type == ClientType.TRANSFORMER:
                client = TransformerClient(
                    id=i,
                    sync_steps=sync_freq,
                    d_z=self.d_z,
                    data_length=self.data_length,
                    y_dim=self.y_dim,
                )
            elif self.client_type == ClientType.ESN:
                client = EchoStateNetworkClient(
                    id=i,
                    sync_steps=sync_freq,
                    d_z=self.d_z,
                    data_length=self.data_length,
                    y_dim=self.y_dim,
                )
            else:
                raise TypeError
            client.set_next_data_sequence(self.data, self.common_target_sequence)
            client.init_p_s(num_clients)
            clients.append(client)
        return clients

    def fit_clients(self, round: int) -> torch.Tensor:
        round_predictions = []
        for client in self.clients:
            next_predictions = client.update_expert(round)
            round_predictions.append(next_predictions)
        return torch.stack(round_predictions).reshape(self.y_dim, self.num_clients)

    def get_predictions_with_beta(self, round: int, betas: torch.Tensor) -> torch.Tensor:
        # beta shape is Nd_z x d_y ---> to N x d_z x d_y
        betas = betas.reshape(self.num_clients, self.d_z, self.y_dim)
        new_round_predictions = []
        i = 0
        for client in self.clients:
            new_pred = client.update_prediction_with_beta(round, betas[i])
            i += 1
            new_round_predictions.append(new_pred)
        return torch.stack(new_round_predictions).reshape(self.y_dim, self.num_clients)

    def get_y(self, t: int) -> torch.Tensor:
        #  All clients have the same target sequence
        return self.common_target_sequence[t].reshape(self.y_dim, 1)
