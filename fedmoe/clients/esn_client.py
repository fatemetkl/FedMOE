from typing import Optional

import torch
import torch.nn as nn

from fedmoe.clients.client import Client
from fedmoe.models.echo_state_net import Esn
from fedmoe.utils.utils import TensorGenerationType

torch.set_default_dtype(torch.float64)


class EchoStateNetworkClient(Client):
    def __init__(
        self,
        id: int,
        sync_steps: int,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: Optional[torch.Tensor] = None,
        affine_map_generator: TensorGenerationType = TensorGenerationType.STANDARD_GAUSSIAN,
    ) -> None:
        self.affine_map_generator = affine_map_generator
        if sigma is None:
            sigma = torch.Tensor([2.0]).repeat(1, y_dim).T
        assert sigma.shape == (y_dim, 1)
        super().__init__(
            id=id, sync_steps=sync_steps, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, alpha=alpha, gamma=gamma, sigma=sigma
        )
        self.latest_Z_T: torch.Tensor

    def init_model(self) -> nn.Module:
        encoder = Esn(
            x_dim=self.x_dim, y_dim=self.y_dim, z_dim=self.z_dim, affine_map_generator=self.affine_map_generator
        )
        return encoder

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        # The input should be a 2D tensor of dimension x_dim x 1.
        assert input.shape == (self.x_dim, 1)
        # Repeating the input by columns to expand into the latent space dimension. Should result in a x_dim x z_dim
        # tensor for encoding.
        input_matrix = input.repeat(1, self.z_dim)
        assert input_matrix.shape == (self.x_dim, self.z_dim)
        return self.encoder(
            input_matrix,
            # Need to subtract one from the current time, since we're predicting for t+1,
            # using \hat{Y}_{t}^i + \beta_{t} Z_{t}, and Z_{t} comes from using Z_{t-1}
            # Concretely, for t=1, \hat{Y}_{1}^i = \hat{Y}_{0}^i + \beta_{0} Z_{0}, and Z_{0} comes from using Z_{-1}
            self.state.get_hidden_state_t(self.state.get_current_time() - 1),
            self.sigma,
        )
