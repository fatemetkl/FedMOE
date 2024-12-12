from dataclasses import dataclass, field
from typing import Set, Tuple

import torch

torch.set_default_dtype(torch.float64)


@dataclass
class GameState:
    _A_times_set: Set[int] = field(default_factory=set)
    _A_hat_times_set: Set[int] = field(default_factory=set)
    _B_times_set: Set[int] = field(default_factory=set)
    _C_times_set: Set[int] = field(default_factory=set)
    _D_times_set: Set[int] = field(default_factory=set)
    _G_times_set: Set[int] = field(default_factory=set)
    _H_times_set: Set[int] = field(default_factory=set)
    _S_time_client_set: Set[Tuple[int, int]] = field(default_factory=set)
    _P_time_client_set: Set[Tuple[int, int]] = field(default_factory=set)
    _D_i_time_client_set: Set[Tuple[int, int]] = field(default_factory=set)
    A: torch.Tensor = field(default_factory=torch.Tensor)
    A_hat: torch.Tensor = field(default_factory=torch.Tensor)
    B: torch.Tensor = field(default_factory=torch.Tensor)
    C: torch.Tensor = field(default_factory=torch.Tensor)
    D: torch.Tensor = field(default_factory=torch.Tensor)
    G: torch.Tensor = field(default_factory=torch.Tensor)
    H: torch.Tensor = field(default_factory=torch.Tensor)

    def clear_state(self) -> None:
        self._A_times_set.clear()
        self._A_hat_times_set.clear()
        self._B_times_set.clear()
        self._C_times_set.clear()
        self._D_times_set.clear()
        self._G_times_set.clear()
        self._H_times_set.clear()
        self._S_time_client_set.clear()
        self._P_time_client_set.clear()
        self._D_i_time_client_set.clear()

    def init_state(self, sync_freq: int, num_clients: int, z_dim: int, y_dim: int) -> None:
        # Go over past time steps and record these values for t in [0, T-1]
        self.A = torch.zeros(
            (sync_freq, num_clients * z_dim, num_clients * z_dim),
        )
        self.A_hat = torch.zeros(
            (sync_freq, num_clients * z_dim, num_clients * z_dim),
        )
        self.B = torch.zeros(
            (sync_freq, num_clients * z_dim, num_clients * y_dim),
        )
        self.C = torch.zeros(
            (sync_freq, num_clients * z_dim, 1),
        )
        self.D = torch.zeros(
            (sync_freq, num_clients * y_dim, num_clients * z_dim),
        )
        self.G = torch.zeros(
            (sync_freq, num_clients * z_dim, num_clients * y_dim),
        )
        self.H = torch.zeros(
            (sync_freq, num_clients * z_dim, 1),
        )
        self.num_clients = num_clients
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.sync_freq = sync_freq

    def set_A_t(self, t: int, A_t: torch.Tensor) -> None:
        self.A[t] = A_t.reshape(self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        assert t not in self._A_times_set
        self._A_times_set.add(t)

    def get_A_t(self, t: int) -> torch.Tensor:
        assert t in self._A_times_set
        return self.A[t]

    def set_A_hat_t(self, t: int, A_hat_t: torch.Tensor) -> None:
        self.A_hat[t] = A_hat_t.reshape(self.num_clients * self.z_dim, self.num_clients * self.z_dim)
        assert t not in self._A_hat_times_set
        self._A_hat_times_set.add(t)

    def get_A_hat_t(self, t: int) -> torch.Tensor:
        assert t in self._A_hat_times_set
        return self.A_hat[t]

    def set_B_t(self, t: int, B_t: torch.Tensor) -> None:
        assert B_t.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        self.B[t] = B_t
        assert t not in self._B_times_set
        self._B_times_set.add(t)

    def get_B_t(self, t: int) -> torch.Tensor:
        assert t in self._B_times_set
        return self.B[t]

    def set_C_t(self, t: int, C_t: torch.Tensor) -> None:
        assert C_t.shape == (self.num_clients * self.z_dim, 1)
        self.C[t] = C_t
        assert t not in self._C_times_set
        self._C_times_set.add(t)

    def get_C_t(self, t: int) -> torch.Tensor:
        assert t in self._C_times_set
        return self.C[t]

    def set_D_t(self, t: int, D_t: torch.Tensor) -> None:
        assert D_t.shape == (self.num_clients * self.y_dim, self.num_clients * self.z_dim)
        self.D[t] = D_t
        assert t not in self._D_times_set
        self._D_times_set.add(t)

    def get_D_t(self, t: int) -> torch.Tensor:
        assert t in self._D_times_set
        return self.D[t]

    def set_G_t(self, t: int, G_t: torch.Tensor) -> None:
        assert G_t.shape == (self.num_clients * self.z_dim, self.num_clients * self.y_dim)
        self.G[t] = G_t
        assert t not in self._G_times_set
        self._G_times_set.add(t)

    def get_G_t(self, t: int) -> torch.Tensor:
        assert t in self._G_times_set
        return self.G[t]

    def set_H_t(self, t: int, H_t: torch.Tensor) -> None:
        assert H_t.shape == (self.num_clients * self.z_dim, 1)
        self.H[t] = H_t
        assert t not in self._H_times_set
        self._H_times_set.add(t)

    def get_H_t(self, t: int) -> torch.Tensor:
        assert t in self._H_times_set
        return self.H[t]

    def print_state(self, t: int) -> str:
        return f"time t (A={self.A[t]}, A_hat={self.A_hat[t]}, B={self.B[t]}, C={self.C[t]}"
