import torch
import torch.nn as nn

from fedmoe.clients.client import Client
from fedmoe.models.random_feature_net import RFN


class RandomFeatureNetworkClient(Client):
    def __init__(
        self,
        id: int,
        sync_steps: int,
        d_z: int,
        data_length: int,
        y_dim: int = 1,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: float = 2,
    ) -> None:
        super().__init__(id, sync_steps, d_z, data_length, y_dim, alpha, gamma, sigma)

    def init_model(self) -> nn.Module:
        A = torch.rand(self.d_z, self.y_dim).double()
        b = torch.rand(self.d_z).double()
        encoder = RFN(A, b)
        return encoder

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input, self.sigma)
