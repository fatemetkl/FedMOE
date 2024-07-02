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
        sigma: float = 0.001,
    ) -> None:
        super().__init__(id, sync_steps, d_z, data_length, y_dim, alpha, gamma, sigma)

    def init_model(self) -> nn.Module:
        A = torch.randn((self.y_dim, self.y_dim)).double()
        b = torch.randn((self.y_dim, self.d_z)).double()
        encoder = RFN(A, b)
        return encoder

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        input_matrix = input.repeat(1, self.d_z)
        return self.encoder(input_matrix, torch.Tensor([self.sigma]).reshape(self.y_dim, 1))
