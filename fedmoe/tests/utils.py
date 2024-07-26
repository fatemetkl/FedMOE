from typing import Tuple

import torch
import torch.nn as nn

from experiments.utils import load_data
from fedmoe.client_manager import ClientManager, ClientType, PreTrainingClientManager
from torch.utils.data import DataLoader
from fedmoe.datasets.periodic_dataset import load_periodic_dataloader
from fedmoe.clients.transformer_client import TransformerClient


def get_data_and_target_sequences() -> Tuple[torch.Tensor, torch.Tensor]:
    # Setup the data and targets
    x_0_sequence = torch.arange(0.1, 1.1, 0.1)
    x_1_sequence = torch.arange(0.2, 2.2, 0.2)
    # x_dim = 2
    data_sequence = torch.stack([x_0_sequence, x_1_sequence], dim=1)
    y_0_sequence = x_0_sequence + x_1_sequence
    y_1_sequence = x_0_sequence * x_0_sequence + x_1_sequence
    y_2_sequence = x_0_sequence + 3 * x_1_sequence
    # y_dim = 3
    target_sequence = torch.stack([y_0_sequence, y_1_sequence, y_2_sequence], dim=1)
    return data_sequence, target_sequence


def get_esn_client_manager(
    alpha: float, gamma: float, sigma: torch.Tensor, z_dim: int, num_clients: int = 2
) -> ClientManager:

    data_sequence, target_sequence = get_data_and_target_sequences()

    # Set seed for reproducibility
    torch.manual_seed(42)

    client_manager = ClientManager(
        ClientType.ESN,
        num_clients,
        data_sequence,
        sync_freq=3,
        z_dim=z_dim,
        alpha=alpha,
        gamma=gamma,
        sigma=sigma,
        target_sequence=target_sequence,
    )

    # Patching the initial conditions with random values to make calculations more complex
    for client in client_manager.clients:
        init_hidden_state_neg1 = torch.rand((3, z_dim))
        init_prediction_0 = torch.rand((3, 1))
        init_prediction_neg1 = torch.rand((3, 1))
        client.state.Z_neg1 = init_hidden_state_neg1
        client.state.Y_0 = init_prediction_0
        client.state.Y_neg1 = init_prediction_neg1
        client.state._predictions[0] = init_prediction_0

    return client_manager


def get_rfn_client_manager(
    alpha: float, gamma: float, sigma: torch.Tensor, z_dim: int, num_clients: int = 2
) -> ClientManager:

    data_sequence, target_sequence = get_data_and_target_sequences()

    # Set seed for reproducibility
    torch.manual_seed(42)

    client_manager = ClientManager(
        ClientType.RFN,
        num_clients,
        data_sequence,
        sync_freq=3,
        z_dim=z_dim,
        alpha=alpha,
        gamma=gamma,
        sigma=sigma,
        target_sequence=target_sequence,
    )

    # Patching the initial conditions with random values to make calculations more complex
    for client in client_manager.clients:
        init_hidden_state_neg1 = torch.rand((3, z_dim))
        init_prediction_0 = torch.rand((3, 1))
        init_prediction_neg1 = torch.rand((3, 1))
        client.state.Z_neg1 = init_hidden_state_neg1
        client.state.Y_0 = init_prediction_0
        client.state.Y_neg1 = init_prediction_neg1
        client.state._predictions[0] = init_prediction_0

    return client_manager


def get_rfn_client_manager_dy_dx_1(alpha: float, gamma: float, z_dim: int, num_clients: int = 2) -> None:
    # Set seed for reproducibility
    torch.manual_seed(42)

    data_sequence = load_data("periodic_signal", 10)

    client_manager = ClientManager(
        ClientType.RFN,
        num_clients,
        data_sequence,
        sync_freq=3,
        z_dim=z_dim,
        alpha=alpha,
        gamma=gamma,
        sigma=1.0,
        target_sequence=None,
    )

    # Patching the initial conditions with random values to make calculations more complex
    for client in client_manager.clients:
        init_hidden_state_neg1 = torch.rand((1, z_dim))
        init_prediction_0 = torch.rand((1, 1))
        init_prediction_neg1 = torch.rand((1, 1))
        client.state.Z_neg1 = init_hidden_state_neg1
        client.state.Y_0 = init_prediction_0
        client.state.Y_neg1 = init_prediction_neg1
        client.state._predictions[0] = init_prediction_0

    return client_manager


class TransformerTestModel(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int) -> None:
        super().__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.linear_1 = torch.nn.Linear(x_dim, 4, bias=False)
        self.linear_2 = torch.nn.Linear(4, y_dim * z_dim, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.linear_1(input.T)
        return self.linear_2(outputs).reshape(self.y_dim, self.z_dim)
    
# Helper function for transformer tests
def get_pre_training_data() -> DataLoader:
    train_dataloader, _, _ = load_periodic_dataloader(
        train_data_size=200,
        val_data_size=200,
        batch_size=5,
        data_length=10 + 1,
    )
    return train_dataloader

def init_model_patch(self) -> nn.Module:
    # x_dim = 2, y_dim = 3, z_dim = 5
    return TransformerTestModel(2, 3, 5)

def get_transformer_client_manager(z_dim: int, sync_freq: int = 3) -> PreTrainingClientManager:

    # Monkey patch the init_model function to bypass pre-training and just return a simple network in the
    # TransformerClient to make life easier
    TransformerClient.init_model = init_model_patch

    data_sequence, target_sequence = get_data_and_target_sequences()

    data_loader = get_pre_training_data()
    client_manager = PreTrainingClientManager(
        num_clients=2,
        data_sequence=data_sequence,
        sync_freq=sync_freq,
        z_dim=z_dim,
        alpha=1.5,
        gamma=2.0,
        pre_training_dataloader=data_loader,
        pre_training_epochs=1,
        pre_training_learning_rate=0.1,
        target_sequence=target_sequence,
    )

    # Patching the initial conditions with random values to make calculations more complex
    for client in client_manager.clients:
        init_hidden_state_neg1 = torch.rand((3, z_dim))
        init_prediction_0 = torch.rand((3, 1))
        init_prediction_neg1 = torch.rand((3, 1))
        client.state.Z_neg1 = init_hidden_state_neg1
        client.state.Y_0 = init_prediction_0
        client.state.Y_neg1 = init_prediction_neg1
        client.state._predictions[0] = init_prediction_0

    return client_manager
