import math

import pytest
import torch

from experiments.utils import load_data
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.game.transformer_game import TransformerGame
from fedmoe.metrics.metrics import RMSEMetric
from fedmoe.server import Server
from fedmoe.tests.utils import (
    get_data_and_target_sequences,
    get_transformer_client_manager,
    setup_transformer_structure_patch,
)


torch.set_default_dtype(torch.float64)

Z_DIM = 3
T = 3


def test_transformer_training_data_metrics() -> None:
    total_rounds = 5
    num_samples = 6
    batch_size = 2
    data_object = load_data("sine_signal", total_rounds + 1)
    train_data_loader = data_object.get_dataloader(num_samples=num_samples, batch_size=batch_size, shuffle=True)
    assert len(train_data_loader) == (num_samples / batch_size)
    for inputs, targets in train_data_loader:
        assert torch.allclose(data_object.input_matrix[:-1], inputs, rtol=0.0, atol=1e-5)
        assert torch.allclose(data_object.target_matrix[1:], targets, rtol=0.0, atol=1e-5)


def test_transformer_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(42)
    num_rounds = 5
    _, target_sequence = get_data_and_target_sequences()

    monkeypatch.setattr(
        TransformerClient,
        "setup_transformer_structure",
        setup_transformer_structure_patch,
    )
    client_manager = get_transformer_client_manager(Z_DIM)
    game = TransformerGame(client_manager.clients, sync_freq=T, z_dim=Z_DIM)
    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[RMSEMetric("RMSE")],
    )
    metric_value = server.fit(num_rounds=num_rounds, have_sync=False)
    loss = torch.Tensor([0.0])
    for time in range(num_rounds + 1):
        loss_value = torch.sum(
            torch.pow(
                torch.subtract(
                    server.server_outputs[time].squeeze(-1).detach(),
                    torch.Tensor(target_sequence[time]),
                ),
                2,
            )
        )
        loss += loss_value
    sqrt_loss = torch.sqrt(torch.div(loss, (num_rounds + 1)))
    metric_pow_2 = metric_value["server - server_predictions - RMSE"] ** 2
    assert math.isclose(sqrt_loss, metric_pow_2, rel_tol=0.0, abs_tol=0.3)
