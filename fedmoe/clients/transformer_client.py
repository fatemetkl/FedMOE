import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from fedmoe.clients.client import Client
from fedmoe.metrics.metrics import MSEMetric
from fedmoe.models.transformer import TransformerTimeSeriesModel


torch.set_default_dtype(torch.float64)


class TransformerClient(Client):
    def __init__(
        self,
        id: int,
        sync_steps: int,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: torch.Tensor | None = None,
        pre_training_epochs: int = 3,
        pre_training_learning_rate: float = 0.01,
        pre_training_dataloader: DataLoader | None = None,
    ) -> None:
        self.pre_training_epochs = pre_training_epochs
        self.pre_training_learning_rate = pre_training_learning_rate
        self.pre_training_dataloader = pre_training_dataloader

        # If we want to do pre_training, then we need to have a data loader.
        if self.pre_training_epochs > 0:
            assert self.pre_training_dataloader is not None
        if sigma is None:
            sigma = torch.Tensor([])
        super().__init__(
            id=id,
            sync_steps=sync_steps,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            alpha=alpha,
            gamma=gamma,
            sigma=sigma,
        )

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        # The input should be a 2D tensor of dimension x_dim x 1.
        assert input.shape == (self.x_dim, 1)
        # Transforming a single input into  (batch_size, seq_len=1, input_dim)
        input_batch = input.T.unsqueeze(0)
        self.encoder.eval()
        # The encoder model should take in a 2D tensor (single time step) of shape (x_dim, 1) and output
        # a tensor of shape y_dim x z_dim
        # We squeeze to remove the extra batch dimension.
        # Shape that the model accepts is (batch_size, seq_len=1, input_dim).
        # Shape that the model outputs is (batch_size, seq_len=1, output_dim).
        return self.encoder(input_batch).squeeze(0).reshape(self.y_dim, self.z_dim)

    @staticmethod
    def pre_train_model(
        model: nn.Module,
        pre_training_epochs: int,
        pre_training_dataloader: DataLoader | None,
        pre_training_learning_rate: float,
        client_id: int,
    ) -> nn.Module:
        pre_training_criterion = nn.MSELoss()
        # If you want to change the metric to MSE make sure to remove the sqrt from the loss function.
        pre_training_metric = MSEMetric("MSE")
        optimizer = optim.Adam(model.parameters(), lr=pre_training_learning_rate, weight_decay=0.002)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        if pre_training_epochs > 0:
            assert pre_training_dataloader is not None
            model.train()
            for epoch in range(0, pre_training_epochs):
                pre_training_metric.clear()
                # For transformer training, target at time t given input t is prediction of {t+1}
                for inputs, targets in pre_training_dataloader:
                    optimizer.zero_grad()

                    outputs = model(inputs, pre_training=True)

                    loss = pre_training_criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    # The outputs and targets here are batch-first, therefore each one is a 3D tensor.
                    # Metrics only accepts up to 2D, so we have to reshape these tensors
                    pre_training_metric.update(
                        outputs.reshape((outputs.size(0), outputs.size(1) * outputs.size(2))),
                        targets.reshape((targets.size(0), targets.size(1) * targets.size(2))),
                    )
                scheduler.step()
                print(
                    f"Transformer pre-training phase: {pre_training_metric.name} client {client_id}\
                        results at epoch {epoch}: {pre_training_metric.compute()}"
                )
        return model

    @staticmethod
    def validate_model(
        encoder: nn.Module,
        validation_sequence: torch.Tensor,
        validation_target: torch.Tensor,
    ) -> dict[str, float]:
        encoder.eval()
        validation_metric = MSEMetric("Validation MSE")
        seq_len = validation_sequence.size(0)

        with torch.no_grad():
            for t in range(seq_len):
                input_t = validation_sequence[t].reshape(-1, 1)  # Shape:(x_dim, 1)
                input_batch = input_t.T.unsqueeze(0)  # Shape (1 , 1, x_dim) -> (batch_size, seq_len=1, input_dim)
                pred_t = encoder(input_batch, pre_training=True).squeeze(
                    0
                )  # Output shape is (1, 1, y_dim) --> (1, y_dim)
                validation_metric.update(
                    pred_t.T, validation_target[t].reshape(-1, 1)
                )  # We store each prediction in shape (y_dim, 1)

            return validation_metric.compute()

    def setup_transformer_structure(self, x_dim: int, y_dim: int, z_dim: int) -> nn.Module:
        # Hyperparameters
        input_dim = x_dim
        hidden_dim = z_dim
        nhead = 2  # Number of heads in multihead attention
        num_encoder_layers = 2  # Number of encoder layers
        dim_feedforward = 16  # Dimension of the feedforward network model
        output_dim = y_dim
        # In this model setup, hidden_dim has the shape d_y times d_z.
        assert self.y_dim * hidden_dim % nhead == 0, (
            "Error: embed_dim (self.y_dim*hidden_dim) must be divisible by num_heads, now it is "
            f"{self.y_dim * hidden_dim} % {nhead}"
        )
        # Create the model
        return TransformerTimeSeriesModel(
            input_dim,
            hidden_dim,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            output_dim,
        )

    def init_model(self) -> nn.Module:
        model = self.setup_transformer_structure(self.x_dim, self.y_dim, self.z_dim)
        # Do pre-training: each client trains its model separately
        return TransformerClient.pre_train_model(
            model,
            self.pre_training_epochs,
            self.pre_training_dataloader,
            self.pre_training_learning_rate,
            self.id,
        )
