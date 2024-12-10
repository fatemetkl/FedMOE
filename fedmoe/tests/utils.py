from typing import Tuple

import torch
import torch.nn as nn

from experiments.utils import load_data
from fedmoe.client_manager import ClientManager, ClientType, PreTrainingClientManager
from fedmoe.clients.transformer_client import TransformerClient

torch.set_default_dtype(torch.float64)


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
    alpha: float,
    gamma: float,
    sigma: torch.Tensor,
    z_dim: int,
    num_clients: int = 2,
    sync_freq: int = 3,
) -> ClientManager:

    data_sequence, target_sequence = get_data_and_target_sequences()

    # Set seed for reproducibility
    torch.manual_seed(42)

    client_manager = ClientManager(
        ClientType.ESN,
        num_clients,
        data_sequence,
        sync_freq=sync_freq,
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


def get_rfn_client_manager_dy_dx_1(
    alpha: float,
    gamma: float,
    sigma: torch.Tensor,
    z_dim: int,
    num_clients: int = 2,
    data_length: int = 10,
    sync_freq: int = 3,
) -> ClientManager:
    # Set seed for reproducibility
    torch.manual_seed(42)

    data_object = load_data("periodic_signal", data_length)

    client_manager = ClientManager(
        ClientType.RFN,
        num_clients,
        data_object.input_matrix,
        sync_freq=sync_freq,
        z_dim=z_dim,
        alpha=alpha,
        gamma=gamma,
        sigma=sigma,
        target_sequence=data_object.target_matrix,
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
        self.linear_1 = torch.nn.Linear(x_dim, 4, bias=False).double()
        self.linear_2 = torch.nn.Linear(4, y_dim * z_dim, bias=False).double()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.linear_1(input.double())
        return self.linear_2(outputs).reshape(self.y_dim, self.z_dim)


def setup_transformer_structure_patch(self, x_dim: int, y_dim: int, z_dim: int) -> nn.Module:  # type: ignore
    return TransformerTestModel(x_dim, y_dim, z_dim)


def get_transformer_client_manager(
    z_dim: int,
    sync_freq: int = 3,
    data_sequence: torch.Tensor | None = None,
    target_sequence: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 1.5,
) -> PreTrainingClientManager:
    # Monkey patch the setup_transformer_structure function to bypass pre-training and just
    # return a simple network in the TransformerClient to make life easier
    if data_sequence is None or target_sequence is None:
        # For the default example we have x_dim = 2, y_dim = 3, z_dim = 5.
        TransformerClient.setup_transformer_structure = setup_transformer_structure_patch  # type: ignore
        data_sequence, target_sequence = get_data_and_target_sequences()
    else:
        TransformerClient.setup_transformer_structure = setup_transformer_structure_patch  # type: ignore

    # y_dim = target_sequence.shape[1]

    client_manager = PreTrainingClientManager(
        num_clients=2,
        data_sequence=data_sequence,
        sync_freq=sync_freq,
        z_dim=z_dim,
        alpha=alpha,
        gamma=gamma,
        pre_training_dataloader=None,
        pre_training_epochs=0,  # Setting pre_training_epochs to zero ensures we do not pre-train the transformer
        pre_training_learning_rate=0.1,
        target_sequence=target_sequence,
    )

    # Patching the initial conditions with random values to make calculations more complex
    # for client in client_manager.clients:
    # init_hidden_state_neg1 = torch.rand((y_dim, z_dim))
    # init_prediction_0 = torch.rand((y_dim, 1))
    # init_prediction_neg1 = torch.rand((y_dim, 1))
    # client.state.Z_neg1 = init_hidden_state_neg1
    # client.state.Y_0 = init_prediction_0
    # client.state.Y_neg1 = init_prediction_neg1
    # client.state._predictions[0] = init_prediction_0

    return client_manager


def print_clients_state(client_manager: ClientManager) -> None:
    for client in client_manager.clients:
        print("client id:", client.id, "---->", repr(client.state))
