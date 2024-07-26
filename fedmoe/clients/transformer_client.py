from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fedmoe.clients.client import Client
from fedmoe.metrics import MSEMetric
from fedmoe.models.transformer import TransformerTimeSeriesModel


class TransformerClient(Client):

    def __init__(
        self,
        id: int,
        sync_steps: int,
        pre_training_dataloader: DataLoader,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: Optional[torch.Tensor] = None,
        pre_training_epochs: int = 3,
        pre_training_learning_rate: float = 0.01,
    ) -> None:
        self.pre_training_epochs = pre_training_epochs
        self.pre_training_learning_rate = pre_training_learning_rate
        self.pre_training_dataloader = pre_training_dataloader
        if sigma is None:
            sigma = torch.Tensor([])
        super().__init__(
            id=id, sync_steps=sync_steps, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, alpha=alpha, gamma=gamma, sigma=sigma
        )

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        # The input should be a 2D tensor of dimension x_dim x 1.
        assert input.shape == (self.x_dim, 1)
        self.encoder.eval()
        # Create a batch-first sample with batch size of 1 for inference
        return self.encoder(input)

    def pre_train_model(self, model: nn.Module) -> nn.Module:
        self.pre_training_criterion = nn.MSELoss()
        self.pre_training_metric = MSEMetric("MSE")
        optimizer = optim.Adam(model.parameters(), lr=self.pre_training_learning_rate)

        model.train()
        for epoch in range(0, self.pre_training_epochs):
            self.pre_training_metric.clear()
            for inputs, targets in self.pre_training_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs.double(), pre_training=True)
                loss = self.pre_training_criterion(outputs.double(), targets.double())
                loss.backward()
                optimizer.step()
                # The outputs and targets here are batch-first, therefore each one is a 3D tensor.
                # Metrics only accepts up to 2D, so we have to reshape these tensors
                self.pre_training_metric.update(
                    outputs.reshape((outputs.size(0), outputs.size(1) * outputs.size(2))),
                    targets.reshape((targets.size(0), targets.size(1) * targets.size(2))),
                )
            print(
                f"Transformer pre-training phase: {self.pre_training_metric.name} client {self.id}\
                    results at epoch {epoch}: {self.pre_training_metric.compute()}"
            )
        return model

    def init_model(self) -> nn.Module:
        # 1) Do pre-training: each client trains its model separately
        # Hyperparameters
        input_dim = self.x_dim
        hidden_dim = self.z_dim
        nhead = 4  # Number of heads in multihead attention
        num_encoder_layers = 3  # Number of encoder layers
        dim_feedforward = 128  # Dimension of the feedforward network model
        output_dim = self.y_dim
        assert hidden_dim % nhead == 0, "Error: embed_dim must be divisible by num_heads"
        # Create the model
        model = TransformerTimeSeriesModel(
            input_dim,
            hidden_dim,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            output_dim,
        )
        return self.pre_train_model(model)
