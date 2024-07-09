from typing import List

import torch
from fl4health.utils.dataset import BaseDataset
from torch.utils.data import DataLoader

from fedmoe.clients.client import Client, ClientType
from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.clients.rfn_client import RandomFeatureNetworkClient
from fedmoe.clients.transformer_client import TransformerClient


class ClientManager:

    def __init__(
        self,
        client_type: ClientType,
        num_clients: int,
        data_sequence: torch.Tensor,
        sync_freq: int,
        d_z: int,
        alpha: float,
        gamma: float,
        sigma: float,
    ) -> None:
        self.client_type = client_type
        self.num_clients = num_clients
        self.d_z = d_z
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.y_dim = 1
        self.data = data_sequence
        self.common_target_sequence = self.set_target()
        self.clients = self.initiate_clients(sync_freq)
        self.sync_freq = sync_freq

    def set_target(self) -> torch.Tensor:
        assert self.data is not None, " data should be set first"
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        return self.data[1:]

    def initiate_clients(self, sync_freq: int) -> List[Client]:
        clients: List[Client] = []
        #  Every tensor type is float64
        for i in range(self.num_clients):
            client: RandomFeatureNetworkClient | EchoStateNetworkClient
            if self.client_type == ClientType.RFN.value:
                client = RandomFeatureNetworkClient(
                    id=i,
                    sync_steps=sync_freq,
                    d_z=self.d_z,
                    y_dim=self.y_dim,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    sigma=self.sigma,
                )
            elif self.client_type == ClientType.ESN.value:
                client = EchoStateNetworkClient(
                    id=i,
                    sync_steps=sync_freq,
                    d_z=self.d_z,
                    y_dim=self.y_dim,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    sigma=self.sigma,
                )
            else:
                raise TypeError
            client.set_next_data_sequence(self.data, self.common_target_sequence)
            client.init_p_s(self.num_clients)
            clients.append(client)
        return clients

    def fit_clients(self, round: int) -> torch.Tensor:
        round_predictions = []
        for client in self.clients:
            t_prediction = client.update_expert(round)
            round_predictions.append(t_prediction)
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

    def update_past_predictions(self, current_round: int, betas: List[torch.Tensor]) -> None:
        # Update client predictions for past T time steps (not including the current time step)
        past_time_steps = [current_round - t for t in range(1, self.sync_freq)]
        round_counter = -1
        for prev_round in past_time_steps:
            round_counter -= 1
            round_betas = betas[round_counter].reshape(self.num_clients, self.d_z, self.y_dim)
            client_counter = 0
            for client in self.clients:
                client.update_prediction_with_beta(prev_round, round_betas[client_counter])
                client_counter += 1

    def get_y(self, t: int) -> torch.Tensor:
        #  All clients have the same target sequence
        return self.common_target_sequence[t].reshape(self.y_dim, 1)

    def get_Y_0(self) -> torch.Tensor:
        init_Y_0 = []
        for client in self.clients:
            init_Y_0_client = client.state.Y_0
            init_Y_0.append(init_Y_0_client)
        return torch.stack(init_Y_0).reshape(self.y_dim, self.num_clients)


class PreTrainingClientManager(ClientManager):

    def __init__(
        self,
        client_type: ClientType,
        num_clients: int,
        data_sequence: BaseDataset,
        sync_freq: int,
        d_z: int,
        alpha: float,
        gamma: float,
        pre_training_dataloader: DataLoader,
        pre_training_epochs: int = 3,
        pre_training_learning_rate: float = 0.01,
    ) -> None:
        self.pre_training_epochs = pre_training_epochs
        self.pre_training_learning_rate = pre_training_learning_rate
        self.pre_training_dataloader = pre_training_dataloader
        # The pre-trained transformer does not have a sigma parameter, so it is initialized to 0.0.
        super().__init__(client_type, num_clients, data_sequence, sync_freq, d_z, alpha, gamma, sigma=0.0)

    def initiate_clients(self, sync_freq: int) -> List[Client]:
        clients: List[Client] = []
        assert self.client_type == ClientType.TRANSFORMER.value, "Error: client should be a transformer based client"
        for i in range(self.num_clients):
            client = TransformerClient(
                id=i,
                sync_steps=sync_freq,
                d_z=self.d_z,
                pre_training_dataloader=self.pre_training_dataloader,
                y_dim=self.y_dim,
                alpha=self.alpha,
                gamma=self.gamma,
                sigma=0.0,  # The pre-trained transformer does not have a sigma parameter
                pre_training_epochs=self.pre_training_epochs,
                pre_training_learning_rate=self.pre_training_learning_rate,
            )
            client.set_next_data_sequence(self.data, self.common_target_sequence)
            client.init_p_s(self.num_clients)
            clients.append(client)
        return clients
