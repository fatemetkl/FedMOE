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
    _current_time: int = -1
    max_time: int = 0
    Z_0: torch.Tensor = torch.Tensor([0.0])

    def get_current_time(self) -> int:
        return self._current_time

    def get_hidden_state_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this hidden state value is not set yet"
        return self._hidden_states[time]

    def get_prediction_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this prediction value is not set yet"
        return self._predictions[time]

    def get_beta_t(self, time: int) -> torch.Tensor:
        assert time <= self._current_time, "Error: this beta value is not set yet"
        return self._betas[time]

    def init_state(self, z: torch.Tensor, Y: torch.Tensor, data_length: int) -> None:
        self._hidden_states.append(z)
        self._predictions.append(Y)
        self.max_time = data_length
        self.Z_0 = z

    def set_beta(self, beta: torch.Tensor, time: int) -> None:
        # Add new beta
        assert self._current_time == time, "Error: time is not the same as current time, check time steps"
        self._betas.append(beta)
        assert (len(self._betas) - 1) == self._current_time

    def next_time_step(self, next_time: int) -> None:
        assert next_time == (self._current_time + 1)
        self._current_time += 1

    def set_next_hidden_state(self, z: torch.Tensor, current_time: int) -> None:
        assert current_time == self._current_time
        self._hidden_states.append(z)

    def set_prediction(self, Y: torch.Tensor, time: int) -> None:
        assert time == self._current_time
        self._predictions.append(Y)

    def replace_prediction_t(self, new_pred: torch.Tensor, time: int) -> None:
        assert 0 <= time <= self._current_time, "Error: this prediction value is not set yet"
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
        self._current_time = -1


class Client(ABC):

    def __init__(
        self,
        id: int,
        sync_steps: int,
        d_z: int,
        data_length: int,
        y_dim: int = 1,
        alpha: float = 0.01,
        gamma: float = 0.1,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.id = id
        self.d_z = d_z
        self.sync_steps = sync_steps
        self.data_length = data_length
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
            init_hidden_state = torch.randn(self.d_z)  # Z shape is: d_z
        if init_prediction is None:
            init_prediction = torch.randn(self.y_dim)
        assert init_hidden_state is not None and init_prediction is not None
        self.state.init_state(init_hidden_state, init_prediction, data_length=self.data_length)

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
        return torch.nn.functional.one_hot(torch.tensor(self.id), num_clients).double()

    def compute_X_t(self, t: int) -> torch.Tensor:
        X = []
        # t-T is the distance from t to the previous sync point (previous T).
        lower_bound = max(t - self.sync_steps, 0)
        for s in range(lower_bound, t + 1):
            X.append(
                torch.mul(
                    pow(math.e, -1 * self.alpha * (t - s)),
                    self.state.get_hidden_state_t(s),
                )
            )
        return torch.stack(X)

    def compute_y_t(self, t: int) -> torch.Tensor:
        y = []
        lower_bound = max(t - self.sync_steps, 0)
        for s in range(lower_bound, t + 1):
            y.append(pow(math.e, -1 * self.alpha * (t - s)) * (self._target[s] - self.state.get_prediction_t(s)))
        return torch.stack(y)

    def update_prediction_with_beta(self, t: int, nash_beta: torch.Tensor) -> torch.Tensor:
        # Replace previous beta
        self.state.replace_beta_t(nash_beta, t)
        # Use the previous Z
        # Update prediction based on Z_t and beta_t
        next_prediction = self.state.get_prediction_t((t - 1)).double() + torch.matmul(
            torch.transpose(nash_beta.double(), 0, 1),
            self.state.get_hidden_state_t(t).double(),
        )
        # next_prediction shape: y_dim*1
        self.state.replace_prediction_t(next_prediction, t)
        return next_prediction

    def update_expert(self, t: int) -> torch.Tensor:
        self.state.next_time_step(next_time=t)
        assert self.state.get_current_time() == t, "Error: time step mismatch"
        # Update X_t
        X_t = self.compute_X_t(t)
        y_t = self.compute_y_t(t)
        X_t_T = torch.transpose(X_t, 0, 1)
        identity_matrix = torch.eye(self.d_z, dtype=torch.float64)
        first_term = torch.matmul(X_t_T, X_t) + self.gamma * identity_matrix
        second_term = torch.matmul(torch.inverse(first_term).double(), X_t_T.double())

        beta_t = torch.matmul(second_term.double(), y_t.double())
        # Place beta_t at the t position in beta list
        self.state.set_beta(beta_t, t)

        # Generate Random State and Update Hidden State
        updated_z = self.feed_encoder(self._current_sequence[t].reshape(self.y_dim, 1))
        self.state.set_next_hidden_state(updated_z, current_time=t)

        # Update prediction based on Z_t and beta_t
        next_prediction = self.state.get_prediction_t(t - 1) + torch.matmul(
            torch.transpose(beta_t, 0, 1).double(),
            self.state.get_hidden_state_t(t).double(),
        )

        # next_prediction shape: y_dim*1
        self.state.set_prediction(next_prediction, t)
        return next_prediction
