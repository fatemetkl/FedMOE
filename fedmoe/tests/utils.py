from typing import Tuple

import torch

from fedmoe.client_manager import ClientManager, ClientType


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


def get_client_manager(
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
