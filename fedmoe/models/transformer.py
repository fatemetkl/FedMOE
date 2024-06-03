import torch
import torch.nn as nn


class TransformerTimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        output_dim: int,
    ) -> None:
        super(TransformerTimeSeriesModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        ).double()
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers).double()
        self.linear = nn.Linear(hidden_dim, output_dim).double()

        self.input_projection = nn.Linear(input_dim, hidden_dim).double()

    def forward(self, x_t: torch.Tensor, pre_training: bool = False) -> torch.Tensor:
        x_t = self.input_projection(x_t)
        encoder_output = self.transformer_encoder(x_t)
        # During fine-tuning, the output of the model is the encoder output.
        if pre_training:
            output = self.linear(encoder_output)
        else:
            output = encoder_output
        return output
