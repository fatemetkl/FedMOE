import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

import torch
import torch.nn as nn


class ClientType(Enum):
    RFN = 0
    TRANSFORMER = 1
    ESN = 2


@dataclass
class ClientState:
    _hidden_states: List[torch.Tensor] = field(default_factory=list)
    _predictions: List[torch.Tensor] = field(default_factory=list)
    _betas: List[torch.Tensor] = field(default_factory=list)
    _current_time: int = 0
    max_time: int = 0
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
        assert time <= self._current_time, "Error: this prediction value is not set yet"
        if time == -1:
            return self.Y_neg1
        return self._predictions[time]

    def get_beta_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this beta value is not set yet"
        return self._betas[time]

    def init_state(self, Z_neg1: torch.Tensor, Y_0: torch.Tensor, Y_neg1: torch.Tensor) -> None:
        self.Z_neg1 = Z_neg1
        self.Y_0 = Y_0
        self._predictions.append(Y_0)
        self.Y_neg1 = Y_neg1

    def set_beta(self, beta: torch.Tensor, time: int) -> None:
        # Add new beta
        assert self._current_time == time + 1, "Error: time is not the same as current time, check time steps"
        self._betas.append(beta)
        assert len(self._betas) == self._current_time

    def next_time_step(self, next_time: int) -> None:
        assert next_time == (self._current_time + 1)
        self._current_time += 1

    def set_hidden_state(self, z: torch.Tensor, time: int) -> None:
        assert time == self._current_time - 1  # Just allows for setting (t-1)'s hidden state.
        self._hidden_states.append(z)

    def set_prediction(self, Y: torch.Tensor, time: int) -> None:
        self._predictions.append(Y)

    def replace_prediction_t(self, new_pred: torch.Tensor, time: int) -> None:
        assert 0 <= time <= self._current_time, "Error: this prediction value is not set yet"
        # print("time-----------", time)
        # print("previous prediction:", self._predictions[time])
        # print("replaced with ", new_pred)
        self._predictions[time] = new_pred

    def replace_beta_t(self, new_beta: torch.Tensor, time: int) -> None:
        assert 0 <= time <= self._current_time, "Error: this beta value is not set yet"
        self._betas[time] = new_beta

    def edit_past_prediction(self, Y: torch.Tensor, time: int) -> None:
        assert 0 <= time < self.max_time, "Error: going over data length"
        assert time <= self._current_time, "Error: this prediction value is not set yet"
        self._predictions[time] = Y

    def clear_state(self) -> None:
        self._hidden_states.clear()
        self._predictions.clear()
        self._current_time = 0


class Client(ABC):

    def __init__(
        self,
        id: int,
        sync_steps: int,
        d_z: int,
        y_dim: int,
        alpha: float,
        gamma: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.id = id
        self.d_z = d_z
        self.sync_steps = sync_steps
        self.y_dim = y_dim
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma

        self.state = ClientState()
        self._current_sequence: torch.Tensor
        self._target: torch.Tensor

        self.encoder: nn.Module = self.init_model()

    @abstractmethod
    def init_model(self) -> nn.Module:
        raise NotImplementedError

    def feed_encoder(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input)

    def set_next_data_sequence(self, data_sequence: torch.Tensor, target_sequence: torch.Tensor) -> None:
        self._current_sequence = data_sequence
        self._target = target_sequence
        # clear previous state, and initiate a new state
        self.state.clear_state()
        self.init_client_state()

    def init_client_state(
        self,
        init_hidden_state: Union[torch.Tensor, None] = None,
        init_prediction: Union[torch.Tensor, None] = None,
    ) -> None:
        if init_hidden_state is None:
            # init_hidden_state = torch.randn((self.y_dim, self.d_z))  # Z shape is: d_z --> changed to dy*dz
            # Initializing Z to zero rather than a random value
            init_hidden_state_neg1 = torch.zeros((self.y_dim, self.d_z))
        if init_prediction is None:
            # Initializing with zero rather than a random value
            # init_prediction = torch.randn((self.y_dim, 1))
            init_prediction_0 = torch.zeros((self.y_dim, 1))
            init_prediction_neg1 = torch.zeros((self.y_dim, 1))
        assert (
            init_prediction_0 is not None and init_prediction_neg1 is not None and init_hidden_state_neg1 is not None
        )
        self.state.init_state(init_hidden_state_neg1, init_prediction_0, init_prediction_neg1)

    def init_p_s(self, num_clients: int) -> None:
        self.P = torch.zeros(
            self.sync_steps + 1,
            num_clients * self.y_dim,
            num_clients * self.y_dim,
            dtype=torch.float64,
        )
        self.S = torch.zeros(self.sync_steps + 1, num_clients, self.y_dim, dtype=torch.float64)

    def get_x(self, t: int) -> torch.Tensor:
        return self._current_sequence[t].reshape(self.y_dim, 1)

    def get_y(self, t: int) -> torch.Tensor:
        return self._target[t].reshape(self.y_dim, 1)

    def get_e(self, num_clients: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(self.id), num_clients).unsqueeze(1).double()

    def compute_X_t(self, t: int) -> torch.Tensor:
        X = []
        # t-T is the distance from t to the previous sync point (previous T).
        lower_bound = max(t - self.sync_steps, 0)
        for s in range(lower_bound, t + 1):
            X.append(
                torch.mul(
                    pow(math.e, -1 * self.alpha * ((t - s) / 2)),
                    self.state.get_hidden_state_t(s - 1).transpose(0, 1),
                )
            )
        prev_time_steps = len(X)
        x = torch.stack(X).reshape(self.d_z, prev_time_steps * self.y_dim)
        return x.transpose(0, 1)

    def compute_y_t(self, t: int) -> torch.Tensor:
        y = []
        lower_bound = max(t - self.sync_steps, 0)
        for s in range(lower_bound, t + 1):
            y.append(
                pow(math.e, -1 * self.alpha * ((t - s) / 2))
                * (self._target[s] - self.state.get_prediction_t(s - 1)).transpose(0, 1)
            )

        prev_time_steps = len(y)
        y_bar = torch.stack(y).reshape(1, prev_time_steps * self.y_dim)
        return y_bar.transpose(0, 1)

    def update_prediction_with_beta(self, t: int, nash_beta: torch.Tensor) -> torch.Tensor:
        # Replace previous beta
        self.state.replace_beta_t(nash_beta, t - 1)
        # Use the previous Z
        # Update prediction based on Z_t and beta_t
        next_prediction = self.state.get_prediction_t((t - 1)).double() + torch.matmul(
            self.state.get_hidden_state_t(t - 1).double(), nash_beta.double()
        )
        # next_prediction shape: y_dim*1
        self.state.replace_prediction_t(next_prediction, t)
        return next_prediction

    def update_expert(self, t: int) -> torch.Tensor:
        self.state.next_time_step(next_time=t)
        assert self.state.get_current_time() == t, "Error: time step mismatch"
        # Update X_t
        X_t = self.compute_X_t(t - 1)
        y_t = self.compute_y_t(t - 1)

        X_t_T = torch.transpose(X_t, 0, 1)
        identity_matrix = torch.eye(self.d_z, dtype=torch.float64)
        first_term = torch.matmul(X_t_T, X_t) + self.gamma * identity_matrix
        second_term = torch.matmul(torch.inverse(first_term).double(), X_t_T.double())

        beta_t = torch.matmul(second_term.double(), y_t.double())
        # Place beta_t at the t-1's position in beta list
        self.state.set_beta(beta_t, t - 1)

        # Generate Random State and Update Hidden State
        updated_z = self.feed_encoder(self._current_sequence[t - 1].reshape(self.y_dim, 1))

        self.state.set_hidden_state(updated_z, time=(t - 1))

        # Update prediction based on Z_t and beta_t
        t_prediction = self.state.get_prediction_t(t - 1) + torch.matmul(
            self.state.get_hidden_state_t(t - 1).double(), beta_t.double()
        )

        # next_prediction shape: y_dim*1
        self.state.set_prediction(t_prediction, t)
        return t_prediction
