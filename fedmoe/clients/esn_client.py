import torch
import torch.nn as nn

from fedmoe.clients.client import Client
from fedmoe.models.echo_state_net import ESN


class EchoStateNetworkClient(Client):
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
        self.latest_Z_T: torch.Tensor

    def init_model(self) -> nn.Module:
        A = torch.rand(self.d_z, self.y_dim).double()
        B = torch.rand(self.d_z, self.d_z).double()
        b = torch.rand(self.d_z).double()
        encoder = ESN(A, B, b)
        return encoder

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(
            input,
            self.state.get_hidden_state_t(self.state.get_current_time()),
            self.sigma,
        )

    def get_latest_Z_T(self) -> torch.Tensor:
        # Latest Z_T is used for Monte Carlo estimation as the initial state to save compute
        current_time = self.state.get_current_time()
        remaining_steps = current_time % self.sync_steps
        # Get the latest T hidden state
        self.latest_Z_T = self.state.get_hidden_state_t(time=(current_time - remaining_steps))
        return self.latest_Z_T
