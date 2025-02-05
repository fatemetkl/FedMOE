import argparse
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.utils import load_config, load_data
from fedmoe.clients.transformer_client import TransformerClient
from fedmoe.models.transformer import TransformerTimeSeriesModel

torch.set_default_dtype(torch.float64)


def setup_transformer_structure(x_dim: int, y_dim: int, z_dim: int) -> nn.Module:
    # Hyperparameters
    input_dim = x_dim
    hidden_dim = z_dim
    nhead = 2  # Number of heads in multihead attention
    num_encoder_layers = 2  # Number of encoder layers
    dim_feedforward = 16  # Dimension of the feedforward network model
    output_dim = y_dim
    # In this model setup, hidden_dim has the shape d_y times d_z.
    assert y_dim * hidden_dim % nhead == 0, "Error: embed_dim (self.y_dim*hidden_dim) must be divisible by num_heads"
    # Create the model
    model = TransformerTimeSeriesModel(
        input_dim,
        hidden_dim,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        output_dim,
    )
    return model


def pre_train_transformer(
    client_id: int,
    x_dim: int,
    y_dim: int,
    z_dim: int,
    train_data_loader: DataLoader,
    pre_training_epochs: int,
    pre_training_learning_rate: float,
    val_data: torch.Tensor,
    val_target: torch.Tensor,
) -> Tuple[nn.Module, torch.Tensor]:
    # we use this if in the phase of client creation we turned off pre-training by passing epoch of 0.
    model = setup_transformer_structure(x_dim, y_dim, z_dim)
    model = TransformerClient.pre_train_model(
        model, pre_training_epochs, train_data_loader, pre_training_learning_rate, client_id
    )
    validation_result = TransformerClient.validate_model(model, val_data, val_target)
    return model, validation_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="experiments/transformer_experiments/pre_train_config.yaml",
    )
    parser.add_argument(
        "--random_seed",
        action="store",
        type=int,
        help="Random seed value.",
        default=2024,
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    data_object = load_data(config["data"], config["total_rounds"] + 1)
    train_data_loader = data_object.get_dataloader(
        num_samples=config["data_loader_num_samples"], batch_size=config["data_loader_batch_size"], shuffle=True
    )
    # save_dir = os.makedirs(, exist_ok=True)
    for client_id in range(config["num_clients"]):
        # Creates a new model and pre-trains it.
        pre_trained_model, validation_result = pre_train_transformer(
            client_id,
            x_dim=data_object.x_dim,
            y_dim=data_object.y_dim,
            z_dim=config["hidden_dim"],
            train_data_loader=train_data_loader,
            pre_training_epochs=config["pre_training_epochs"],
            pre_training_learning_rate=config["pre_training_learning_rate"],
            val_data=data_object.input_matrix,
            val_target=data_object.target_matrix,
        )
        model_name = f"client_{client_id}_model.pth"
        model_path = config["models_dir"] + model_name
        torch.save(pre_trained_model.state_dict(), model_path)

        print(f"Validation result client {client_id}:", validation_result)
