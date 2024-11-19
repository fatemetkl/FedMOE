from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ClientState:
    _hidden_states: List[torch.Tensor] = field(default_factory=list)
    _predictions: List[torch.Tensor] = field(default_factory=list)
    _betas: List[torch.Tensor] = field(default_factory=list)
    _current_time: int = -1
    # These represent the a priori values that are required to start the time series modeling. In the notation of the
    # paper, these are Z_^i{-1}, \hat{Y}^i_{0}, \hat{Y}^i_{-1}
    Z_neg1: torch.Tensor = torch.Tensor([0.0])
    Y_0: torch.Tensor = torch.Tensor([0.0])
    Y_neg1: torch.Tensor = torch.Tensor([0.0])

    def get_current_time(self) -> int:
        return self._current_time

    def get_hidden_state_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this hidden state value is not set yet"
        if time == -1:
            return self.Z_neg1
        return self._hidden_states[time]

    def get_prediction_t(self, time: int) -> torch.Tensor:
        # Because at time t (self._current_time) we set the prediction of next step (t+1).
        assert time <= self._current_time + 1, "Error: this prediction value is not set yet"
        if time == -1:
            assert self.Y_neg1.shape == (self.y_dim, 1)
            return self.Y_neg1
        return self._predictions[time]

    def get_beta_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this beta value is not set yet"
        return self._betas[time]

    def init_state(
        self, z_dim: int, y_dim: int, Z_neg1: torch.Tensor, Y_0: torch.Tensor, Y_neg1: torch.Tensor
    ) -> None:
        self.z_dim = z_dim
        self.y_dim = y_dim
        assert Z_neg1.shape == (self.y_dim, self.z_dim)
        self.Z_neg1 = Z_neg1
        assert Y_0.shape == (self.y_dim, 1)
        self.Y_0 = Y_0
        self._predictions.append(Y_0)
        assert Y_neg1.shape == (self.y_dim, 1)
        self.Y_neg1 = Y_neg1

    def set_beta(self, beta: torch.Tensor, time: int) -> None:
        # Add new beta
        assert self._current_time == time, "Error: time is not the same as current time, check time steps"
        assert beta.shape == (self.z_dim, 1)
        self._betas.append(beta)
        assert len(self._betas) == self._current_time + 1

    def next_time_step(self, next_time: int) -> None:
        assert next_time == (self._current_time + 1)
        self._current_time += 1

    def set_hidden_state(self, z: torch.Tensor, time: int) -> None:
        assert time == self._current_time
        assert z.shape == (
            self.y_dim,
            self.z_dim,
        ), f"Error: z's shape is {z.shape}, expected {(self.y_dim, self.z_dim)}, time is {time}"
        self._hidden_states.append(z)

    def set_prediction(self, Y: torch.Tensor, time: int) -> None:
        # check it is not previously set
        assert len(self._predictions) == time, "Error: this prediction value is already set"
        assert Y.shape == (self.y_dim, 1)
        self._predictions.append(Y)

    def replace_prediction_t(self, new_pred: torch.Tensor, time: int) -> None:
        # At each time t, the Y_{t+1} value could be set and replaced.
        assert 0 <= time <= (self._current_time + 1), "Error: this prediction value is not set yet"
        assert new_pred.shape == (self.y_dim, 1)
        self._predictions[time] = new_pred

    def replace_beta_t(self, new_beta: torch.Tensor, time: int) -> None:
        assert 0 <= time <= self._current_time, "Error: this beta value is not set yet"
        assert new_beta.shape == (self.z_dim, 1)
        self._betas[time] = new_beta

    def clear_state(self) -> None:
        self._hidden_states.clear()
        self._predictions.clear()
        self._betas.clear()
        self._current_time = -1

    def __repr__(self) -> str:
        dz_rep = "____hidden_states____\n"
        Y_rep = "____predictions____\n"
        for time in range(self._current_time):
            dz_rep += f"| time {time}: {self._hidden_states[time]}___|\n"

        for time in range(self._current_time + 1):
            Y_rep += f"| time {time}: {self._predictions[time]}___|\n"

        return dz_rep + Y_rep
