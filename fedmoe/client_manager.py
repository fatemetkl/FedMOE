from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from fedmoe.clients.client import Client, ClientType
from fedmoe.clients.esn_client import EchoStateNetworkClient
from fedmoe.clients.rfn_client import RandomFeatureNetworkClient
from fedmoe.clients.transformer_client import TransformerClient

torch.set_default_dtype(torch.float64)


class ClientManager:
    def __init__(
        self,
        client_type: ClientType,
        num_clients: int,
        data_sequence: torch.Tensor,
        sync_freq: int,
        z_dim: int,
        alpha: float,
        gamma: float,
        sigma: Union[float, torch.Tensor],
        target_sequence: Optional[torch.Tensor] = None,
        game_T: Optional[int] = None,
    ) -> None:
        self.client_type = client_type
        self.num_clients = num_clients
        self.z_dim = z_dim
        self.alpha = alpha
        self.gamma = gamma
        # game_T in client manager is only used to initiate P and S matrix sizes.
        # If we don't provide game_T, then it is set to sync_freq (client T).
        # Initiating P and S with client T rather than game T won't be problematic, but
        # in that case P and S will overflow if game_T is larger than client T.
        if game_T is None:
            self.game_T = sync_freq
        else:
            self.game_T = game_T
        # Shape is time x x_dim
        self.data = data_sequence
        assert len(data_sequence.shape) == 2
        self.x_dim = self.data.shape[1]

        # Shape of common_target_sequence is time x y_dim
        if target_sequence is None:
            self.common_target_sequence = self.set_target()
        else:
            self.common_target_sequence = target_sequence

        self.y_dim = self.common_target_sequence.shape[1]

        self.sigma = sigma if isinstance(sigma, torch.Tensor) else torch.Tensor([sigma]).repeat(1, self.y_dim).T
        assert self.sigma.shape == (self.y_dim, 1)

        self.sync_freq = sync_freq
        self.clients = self.initiate_clients()

    def set_target(self) -> torch.Tensor:
        assert self.data is not None, " data should be set first"
        # Target is the shifted input to the left
        #  y_0 = x_1 (predict next input)
        return self.data[1:]

    def initiate_clients(self) -> List[Client]:
        clients: List[Client] = []
        #  Every tensor type is float64
        for i in range(self.num_clients):
            client: RandomFeatureNetworkClient | EchoStateNetworkClient
            if self.client_type == ClientType.RFN:
                client = RandomFeatureNetworkClient(
                    id=i,
                    sync_steps=self.sync_freq,
                    x_dim=self.x_dim,
                    y_dim=self.y_dim,
                    z_dim=self.z_dim,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    sigma=self.sigma,
                )
            elif self.client_type == ClientType.ESN:
                client = EchoStateNetworkClient(
                    id=i,
                    sync_steps=self.sync_freq,
                    x_dim=self.x_dim,
                    y_dim=self.y_dim,
                    z_dim=self.z_dim,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    sigma=self.sigma,
                )
            else:
                raise TypeError
            client.set_next_data_sequence(self.data, self.common_target_sequence)
            client.init_p_s(self.num_clients, self.game_T)
            clients.append(client)
        return clients

    def fit_clients(self, round: int) -> torch.Tensor:
        round_predictions = []
        for client in self.clients:
            t_prediction = client.update_expert(round)
            # Each t_prediction is a column tensor d_y x 1
            round_predictions.append(t_prediction)
        # Shape of returned tensor is  N (number of clients) x d_y
        predictions_tensor = torch.stack(round_predictions, dim=0).squeeze(-1)
        assert predictions_tensor.shape == (self.num_clients, self.y_dim)
        return predictions_tensor

    def improve_previous_predictions_from_game(
        self, round: int, game_predictions: torch.Tensor, beta_t: torch.Tensor
    ) -> None:
        # game_predictions tensor shape is NDy x 1 --->should be reshaped to N x Dy x 1
        game_predictions = game_predictions.reshape(self.num_clients, self.y_dim, 1)
        beta_t = beta_t.reshape(self.num_clients, self.z_dim, 1)
        for client in self.clients:
            client.state.replace_prediction_t(game_predictions[client.id].reshape(-1, 1), round)
            client.state.replace_beta_t(beta_t[client.id], round)

    def get_predictions_with_beta(self, round: int, betas: torch.Tensor) -> torch.Tensor:
        # beta shape is Nd_z x 1 ---> to N x d_z x 1
        betas = betas.reshape(self.num_clients, self.z_dim, 1)
        new_round_predictions = []
        i = 0
        for client in self.clients:
            new_pred = client.update_prediction_with_beta(round, betas[i])
            i += 1
            new_round_predictions.append(new_pred)
        predictions_tensor = torch.stack(new_round_predictions, dim=0).squeeze(-1)
        assert predictions_tensor.shape == (self.num_clients, self.y_dim)
        return predictions_tensor

    def update_past_predictions(self, current_round: int, betas: List[torch.Tensor]) -> None:
        # Update client predictions for past T time steps (not including the current time step)
        past_time_steps = [current_round - t for t in range(1, self.sync_freq)]
        round_counter = -1
        for prev_round in past_time_steps:
            round_counter -= 1
            round_betas = betas[round_counter].reshape(self.num_clients, self.z_dim, self.y_dim)
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
            init_Y_0_client = client.state.get_prediction_t(0)
            init_Y_0.append(init_Y_0_client)
        init_Y_tensor = torch.stack(init_Y_0, dim=0).squeeze(-1)
        assert init_Y_tensor.shape == (
            self.num_clients,
            self.y_dim,
        ), f"Error: Y_0 matrix shape is {init_Y_tensor.shape}"
        return init_Y_tensor


class PreTrainingClientManager(ClientManager):
    def __init__(
        self,
        num_clients: int,
        data_sequence: torch.Tensor,
        sync_freq: int,
        z_dim: int,
        alpha: float,
        gamma: float,
        pre_training_dataloader: Optional[DataLoader] = None,
        pre_training_epochs: int = 3,
        pre_training_learning_rate: float = 0.01,
        target_sequence: Optional[torch.Tensor] = None,
        game_T: Optional[int] = None,
    ) -> None:
        self.pre_training_epochs = pre_training_epochs
        self.pre_training_learning_rate = pre_training_learning_rate
        self.pre_training_dataloader = pre_training_dataloader
        # If we want to do pre_training, then we need to have a data loader.
        if self.pre_training_epochs > 0:
            assert self.pre_training_dataloader is not None
        super().__init__(
            client_type=ClientType.TRANSFORMER,
            num_clients=num_clients,
            data_sequence=data_sequence,
            sync_freq=sync_freq,
            z_dim=z_dim,
            alpha=alpha,
            gamma=gamma,
            # The pre-trained transformer does not have a sigma parameter, so it is initialized to 0.0.
            sigma=0.0,
            target_sequence=target_sequence,
            game_T=game_T,
        )

    def initiate_clients(self) -> List[Client]:
        clients: List[Client] = []
        assert self.client_type == ClientType.TRANSFORMER, "Error: client should be a transformer based client"
        for i in range(self.num_clients):
            client = TransformerClient(
                id=i,
                sync_steps=self.sync_freq,
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                z_dim=self.z_dim,
                alpha=self.alpha,
                gamma=self.gamma,
                sigma=None,  # The pre-trained transformer does not have a sigma parameter
                pre_training_epochs=self.pre_training_epochs,
                pre_training_learning_rate=self.pre_training_learning_rate,
                pre_training_dataloader=self.pre_training_dataloader,
            )
            client.set_next_data_sequence(self.data, self.common_target_sequence)
            client.init_p_s(self.num_clients, self.game_T)
            clients.append(client)
        return clients
