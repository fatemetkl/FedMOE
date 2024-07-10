import torch
import torch.nn as nn


class RFN(nn.Module):

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
    ):
        super(RFN, self).__init__()
        self.A = A
        self.b = b

    def generate_random_state(self, sigma_t: torch.Tensor) -> torch.Tensor:
        W_t = torch.normal(mean=torch.zeros((1, self.b.shape[1])), std=1)
        random_state = torch.matmul(sigma_t, W_t)
        # random state shape: d_y*d_z
        return random_state

    def forward(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        random_state = self.generate_random_state(sigma_t)
        #  X shape: y_dim * 1
        AX = torch.matmul(self.A, x_t.double()).squeeze(1)
        Z = nn.ReLU()(AX + self.b + random_state)
        # Output size should be d_z*d_y
        return Z
