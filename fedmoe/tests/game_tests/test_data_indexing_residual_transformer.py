import torch

from experiments.utils import load_data
from fedmoe.client_manager import PreTrainingClientManager
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.game.transformer_game import TransformerGame
from fedmoe.server import Server
from fedmoe.tests import utils

TOTAL_ROUNDS = 20
data_object = load_data("periodic_signal", TOTAL_ROUNDS + 1)
Z_DIM = 3
T = 5
ALPHA = 0.1
GAMMA = 0.1
Y_DIM = 1


def test_data_indexing(monkeypatch) -> None:
    """
    The goal of this test is to visually confirm and check that T step's prediction with game are good
    and result is a drop in residual.
    """
    monkeypatch.setattr(TransformerClient, "setup_transformer_structure", utils.setup_transformer_structure_patch)
    client_manager = PreTrainingClientManager(
        num_clients=2,
        data_sequence=data_object.input_matrix,
        sync_freq=T,
        z_dim=Z_DIM,
        alpha=ALPHA,
        gamma=GAMMA,
        pre_training_dataloader=None,
        pre_training_epochs=0,  # Setting pre_training_epochs to zero ensures we do not pre-train the transformer
        pre_training_learning_rate=0.1,
        target_sequence=data_object.target_matrix,
    )

    game = TransformerGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=Z_DIM,
    )

    server = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=1.0,
        eta=1.0,
    )
    server_residuals = []
    _ = server.fit(TOTAL_ROUNDS, have_sync=True)

    for t in range(TOTAL_ROUNDS - 1):
        w_t = server.mixture_weights[t]
        next_predictions = server.clients_predictions[t + 1]
        residual = data_object.target_matrix[t + 1].reshape(Y_DIM, 1) - torch.matmul(w_t.T, next_predictions).T
        inner_residual = torch.pow(torch.linalg.norm(residual), 2.0)
        server_residuals.append(inner_residual)
        if t % T == 0 and t > 0:
            print("THIS IS SYNC STEP")
        print(f"server residual {inner_residual}, time {t} predicting {t+1}")

    # assert False
