import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class TransformerTimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        output_dim: int,
        max_seq_len: int = 500,  # Default max sequence length
    ) -> None:
        super(TransformerTimeSeriesModel, self).__init__()
        self.dropout = 0.1
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, output_dim * hidden_dim)

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, output_dim * hidden_dim)
        )  # Learnable positional encoding

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim * hidden_dim,
            nhead=nhead,
            dropout=self.dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Output projection for pre-training
        self.linear = nn.Linear(output_dim * hidden_dim, output_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        pre_training: bool = False,
    ) -> torch.Tensor:
        """
        Args:
        - x_t: Input tensor of shape (batch_size, seq_len, input_dim).
        - pre_training: Whether the model is in pre-training mode (next-step prediction).

        """
        x_t = self.input_projection(x_t)  # Project input to transformer dimension
        seq_len = x_t.size(1)
        x_t = x_t + self.positional_encoding[:, :seq_len]  # Add positional encoding

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # Transformer encoder
        encoder_output = self.transformer_encoder(x_t, mask=causal_mask, is_causal=True)

        # Output projection during pre-training
        if pre_training:
            output = self.linear(encoder_output)  # Predict next value
        else:
            output = encoder_output  # Return encoder outputs for downstream tasks

        return output


# class TransformerTimeSeriesModel(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         nhead: int,
#         num_encoder_layers: int,
#         dim_feedforward: int,
#         output_dim: int,
#     ) -> None:
#         self.dropout = 0.1
#         super(TransformerTimeSeriesModel, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=output_dim*hidden_dim,
#             nhead=nhead,
#             dropout=self.dropout,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
#  num_layers=num_encoder_layers)
#         self.linear = nn.Linear(output_dim * hidden_dim, output_dim)

#         self.input_projection = nn.Linear(input_dim, output_dim*hidden_dim)
#         # self.positional_encoding = nn.Parameter(torch.zeros(1, 15, output_dim * hidden_dim))

#     def forward(self, x_t: torch.Tensor,  attention_mask=None, pre_training: bool = False) -> torch.Tensor:
#         x_t = self.input_projection(x_t)
#         # + self.positional_encoding[:, : x_t.size(1)]
#         encoder_output = self.transformer_encoder(x_t)
#         # During fine-tuning, the output of the model is the encoder output.
#         if pre_training:
#             output = self.linear(encoder_output)
#         else:
#             output = encoder_output
#         return output
