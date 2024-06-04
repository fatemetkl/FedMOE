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
        d_z: int,
        data_length: int,
        pre_training_dataloader: DataLoader,
        y_dim: int = 1,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: float = 2,
        pre_training_epochs: int = 3,
        pre_training_learning_rate: float = 0.01,
    ) -> None:
        super().__init__(id, sync_steps, d_z, data_length, y_dim, alpha, gamma, sigma)
        self.pre_training_epochs = pre_training_epochs
        self.pre_training_learning_rate = pre_training_learning_rate
        self.pre_training_dataloader = pre_training_dataloader

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        # Create a batch-first sample with batch size of 1 for inference
        return self.encoder(input.reshape(1, self.y_dim).double())[0]

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
                loss = self.pre_training_criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                self.pre_training_metric.update(outputs, targets)
            print(
                f"Transformer pre-training phase: {self.pre_training_metric.name} client {self.id}\
                    results at epoch {epoch}: {self.pre_training_metric.compute()}"
            )
        return model

    def init_model(self) -> nn.Module:
        # 1) Do pre-training: each client trains its model seperately
        # Hyperparameters
        input_dim = self.y_dim
        hidden_dim = self.d_z
        nhead = 4  # Number of heads in multihead attention
        num_encoder_layers = 3  # Number of encoder layers
        dim_feedforward = 128  # Dimension of the feedforward network model
        output_dim = self.y_dim
        assert hidden_dim % nhead == 0, "Error: mbed_dim must be divisible by num_heads"
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
