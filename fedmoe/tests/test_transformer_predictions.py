import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedmoe.client_manager import PreTrainingClientManager
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.datasets.periodic_dataset import load_periodic_dataloader
from fedmoe.tests.utils import TransformerTestModel, get_data_and_target_sequences

DATA_SEQUENCE, TARGET_SEQUENCE = get_data_and_target_sequences()
Z_DIM = 5


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
    return TransformerTestModel(2, 3, Z_DIM)


def test_tranformer_client_prediction_process() -> None:

    # Fixing seed for reproducible sampling trajectory
    torch.manual_seed(42)

    # Monkey patch the init_model function to bypass pre-training and just return a simple network in the
    # TransformerClient to make life easier
    TransformerClient.init_model = init_model_patch

    data_loader = get_pre_training_data()
    client_manager = PreTrainingClientManager(
        num_clients=2,
        data_sequence=DATA_SEQUENCE,
        sync_freq=10,
        z_dim=Z_DIM,
        alpha=1.5,
        gamma=2.0,
        pre_training_dataloader=data_loader,
        pre_training_epochs=1,
        pre_training_learning_rate=0.1,
        target_sequence=TARGET_SEQUENCE,
    )

    # Patching the initial conditions with random values to make calculations more complex
    for client in client_manager.clients:
        init_hidden_state_neg1 = torch.rand((3, Z_DIM))
        init_prediction_0 = torch.rand((3, 1))
        init_prediction_neg1 = torch.rand((3, 1))
        client.state.Z_neg1 = init_hidden_state_neg1
        client.state.Y_0 = init_prediction_0
        client.state.Y_neg1 = init_prediction_neg1
        client.state._predictions[0] = init_prediction_0

    # Transforms in the linear model (fixed by the random seed)
    linear_1 = torch.Tensor([[0.5406, 0.5869], [-0.1657, 0.6496], [-0.1549, 0.1427], [-0.3443, 0.4153]]).double()
    linear_2 = torch.Tensor(
        [
            [0.4408, -0.3668, 0.4346, 0.0936],
            [0.3694, 0.0677, 0.2411, -0.0706],
            [0.3854, 0.0739, -0.2334, 0.1274],
            [-0.2304, -0.0586, -0.2031, 0.3317],
            [-0.3947, -0.2305, -0.1412, -0.3006],
            [0.0472, -0.4938, 0.4516, -0.4247],
            [0.3860, 0.0832, -0.1624, 0.3090],
            [0.0779, 0.4040, 0.0547, -0.1577],
            [0.1343, -0.1356, 0.2104, 0.4464],
            [0.2890, -0.2186, 0.2886, 0.0895],
            [0.2539, -0.3048, -0.4950, -0.1932],
            [-0.3835, 0.4103, 0.1440, 0.2071],
            [0.1581, -0.0087, 0.3913, -0.3553],
            [0.0315, -0.3413, 0.1542, -0.1722],
            [0.1532, -0.1042, 0.4147, -0.2964],
        ]
    ).double()

    # Making prediction for t=1
    t = 1
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_0 = torch.Tensor([[0.0050], [0.0757], [-0.1199], [-0.0651], [-0.1122]]).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, nn.Module)
    assert isinstance(client_0, TransformerClient)
    z_0 = client_0.state.get_hidden_state_t(t - 1)

    beta_0_target = client_0.state.get_beta_t(t - 1)
    assert torch.allclose(beta_0, beta_0_target, rtol=0.0, atol=1e-3)
    x_0_matrix = DATA_SEQUENCE[0].reshape(-1, 1).double()
    # y_dim = 3. Also, no idea why but pytorch does the linear algebra weirdly...
    z_0_target = torch.matmul(linear_2, torch.matmul(linear_1, x_0_matrix)).T.reshape(3, Z_DIM)

    assert torch.allclose(z_0_target, z_0.double(), rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_0_target, beta_0) + client_0.state.Y_0
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 2
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_1 = torch.Tensor([[-0.0035], [0.0155], [-0.0216], [-0.0506], [-0.0523]]).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, nn.Module)
    assert isinstance(client_0, TransformerClient)
    z_1 = client_0.state.get_hidden_state_t(t - 1)

    beta_1_target = client_0.state.get_beta_t(t - 1)
    assert torch.allclose(beta_1, beta_1_target, rtol=0.0, atol=1e-3)
    x_1_matrix = DATA_SEQUENCE[1].reshape(-1, 1).double()
    # y_dim = 3. Also, no idea why but pytorch does the linear algebra weirdly...
    z_1_target = torch.matmul(linear_2, torch.matmul(linear_1, x_1_matrix)).T.reshape(3, Z_DIM)

    assert torch.allclose(z_1_target, z_1.double(), rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_1_target, beta_1) + client_0.state.get_prediction_t(t - 1)
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)

    t = 3
    predictions = client_manager.fit_clients(t)
    assert predictions.shape == (3, 2)

    beta_2 = torch.Tensor([[-0.0067], [0.0478], [0.0759], [-0.1007], [-0.0580]]).double()
    client_0 = client_manager.clients[0]
    client_0_encoder = client_0.encoder
    assert isinstance(client_0_encoder, nn.Module)
    assert isinstance(client_0, TransformerClient)
    z_2 = client_0.state.get_hidden_state_t(t - 1)

    beta_2_target = client_0.state.get_beta_t(t - 1)
    assert torch.allclose(beta_2, beta_2_target, rtol=0.0, atol=1e-3)
    x_2_matrix = DATA_SEQUENCE[2].reshape(-1, 1).double()
    # y_dim = 3. Also, no idea why but pytorch does the linear algebra weirdly...
    z_2_target = torch.matmul(linear_2, torch.matmul(linear_1, x_2_matrix)).T.reshape(3, Z_DIM)

    assert torch.allclose(z_2_target, z_2.double(), rtol=0.0, atol=1e-3)

    # Make sure predictions is good as well.
    target_pred = torch.matmul(z_2_target, beta_2) + client_0.state.get_prediction_t(t - 1)
    assert torch.allclose(predictions[:, 0].reshape(-1, 1), target_pred, rtol=0.0, atol=1e-3)
