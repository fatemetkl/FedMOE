import math

import torch
import torch.nn as nn


class ESN(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        b: torch.Tensor,
    ):
        super(ESN, self).__init__()
        self.A = A
        self.B = B
        self.b = b

    def generate_random_state(self, sigma_t: float) -> torch.Tensor:
        W_t = torch.normal(mean=torch.zeros(len(self.b)), std=1)
        random_state = math.sqrt(sigma_t) * W_t
        return random_state

    def forward(self, x_t: torch.Tensor, Z_client_t: torch.Tensor, sigma_t: float) -> torch.Tensor:
        random_state = self.generate_random_state(sigma_t)
        #  X shape: y_dim * 1
        AX = torch.matmul(self.A, x_t.double()).squeeze(1)
        BZ = torch.matmul(self.B, Z_client_t.double())
        Z = nn.ReLU()(AX + BZ + self.b + random_state)
        # Output size should be d_z
        return Z
