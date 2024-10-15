import torch
import torch.nn as nn

from fedmoe.utils.utils import TensorGenerationType, generate_random_tensor


class Rfn(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        affine_map_generator: TensorGenerationType = TensorGenerationType.STANDARD_GAUSSIAN,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.affine_map_generator = affine_map_generator
        self.generate_affine_map()

    def generate_affine_map(self) -> None:
        self.A = generate_random_tensor(self.affine_map_generator, (self.y_dim, self.x_dim))
        self.b = generate_random_tensor(self.affine_map_generator, (self.y_dim, self.z_dim))

    def generate_random_state(self, sigma_t: torch.Tensor) -> torch.Tensor:
        W_t = torch.normal(mean=torch.zeros((1, self.z_dim)), std=1.0)
        random_state = torch.matmul(sigma_t, W_t)
        assert random_state.shape == (self.y_dim, self.z_dim)
        # random state shape: y_dim x z_dim
        return random_state

    def forward(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        # input should be of shape x_dim x z_dim
        assert x_t.shape[1] == self.z_dim

        random_state = self.generate_random_state(sigma_t)

        AX = torch.matmul(self.A, x_t.double())
        #  AX shape should be y_dim x z_dim
        assert AX.shape == (self.y_dim, self.z_dim)

        Z_t = nn.ReLU()(AX + self.b + random_state)
        # Latent space size should be y_dim x z_dim
        assert Z_t.shape == (self.y_dim, self.z_dim)

        return Z_t
