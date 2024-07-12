import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

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
        self._predictions[time] = new_pred

    def replace_beta_t(self, new_beta: torch.Tensor, time: int) -> None:
        assert 0 <= time <= self._current_time, "Error: this beta value is not set yet"
        self._betas[time] = new_beta

    def clear_state(self) -> None:
        self._hidden_states.clear()
        self._predictions.clear()
        self._current_time = 0


class Client(ABC):

    def __init__(
        self,
        id: int,
        sync_steps: int,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        alpha: float,
        gamma: float,
        sigma: torch.Tensor,
    ) -> None:
        super().__init__()
        self.id = id
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.sync_steps = sync_steps
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
        init_hidden_state_neg1: Union[torch.Tensor, None] = None,
        init_prediction_0: Union[torch.Tensor, None] = None,
        init_prediction_neg1: Optional[torch.Tensor] = None,
    ) -> None:
        if init_hidden_state_neg1 is None:
            # Initializing Z to zero rather than a random value
            init_hidden_state_neg1 = torch.zeros((self.y_dim, self.z_dim))
        if init_prediction_0 is None:
            # Initializing with zero rather than a random value
            init_prediction_0 = torch.zeros((self.y_dim, 1))
        if init_prediction_neg1 is None:
            # Initializing with zero rather than a random value
            init_prediction_neg1 = torch.zeros((self.y_dim, 1))
        assert (
            init_prediction_0 is not None and init_prediction_neg1 is not None and init_hidden_state_neg1 is not None
        )
        # Make sure everything is the right shape
        assert init_hidden_state_neg1.shape == (self.y_dim, self.z_dim)
        assert init_prediction_0.shape == (self.y_dim, 1)
        assert init_prediction_neg1.shape == (self.y_dim, 1)
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
        return self._current_sequence[t].reshape(-1, 1)

    def get_y(self, t: int) -> torch.Tensor:
        return self._target[t].reshape(self.y_dim, 1)

    def get_e(self, num_clients: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(self.id), num_clients).unsqueeze(1).double()

    def compute_X_t(self, t: int) -> torch.Tensor:
        X = []
        # t-T is the distance from t to the previous sync point (previous T).
        start_point = max(t - self.sync_steps + 1, 0)
        for s in range(start_point, t + 1):
            X.append(torch.mul(pow(math.e, -1 * self.alpha * ((t - s) / 2)), self.state.get_hidden_state_t(s - 1)))
        return torch.cat(X)

    def compute_y_t(self, t: int) -> torch.Tensor:
        y = []
        start_point = max(t - self.sync_steps + 1, 0)
        for s in range(start_point, t + 1):
            # target has shape time x y_dim so need to transform to column vector after indexing (occurs in get_y)
            residual = self.get_y(s) - self.state.get_prediction_t(s - 1)
            assert residual.shape == (self.y_dim, 1)
            y.append(pow(math.e, -1 * self.alpha * ((t - s) / 2)) * residual)
        return torch.cat(y)

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

    def optimize_beta(self, t: int) -> torch.Tensor:
        # Update X_t
        X_t = self.compute_X_t(t - 1)
        y_t = self.compute_y_t(t - 1)

        X_t_T = torch.transpose(X_t, 0, 1)
        identity_matrix = torch.eye(self.z_dim, dtype=torch.float64)
        first_term = torch.matmul(X_t_T, X_t) + self.gamma * identity_matrix
        second_term = torch.matmul(torch.inverse(first_term).double(), X_t_T.double())
        beta_t = torch.matmul(second_term.double(), y_t.double())
        return beta_t

    def predict(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First optimize betas based on past observations
        beta_t = self.optimize_beta(t)

        # Generate Random State and update Hidden State
        # Sequence has shape time x x_dim, but encoders expect input for a given step of x_dim x 1 (occurs in get_x)
        updated_z = self.feed_encoder(self.get_x(t - 1))

        # Update prediction based on Z_t and beta_t
        t_prediction = self.state.get_prediction_t(t - 1) + torch.matmul(updated_z.double(), beta_t.double())
        return beta_t, updated_z, t_prediction

    def update_expert(self, t: int) -> torch.Tensor:
        self.state.next_time_step(next_time=t)
        assert self.state.get_current_time() == t, "Error: time step mismatch"

        # Make prediction for time t
        beta_t, updated_z, t_prediction = self.predict(t)

        # Update State
        # Place beta_t at the t-1's position in beta list
        self.state.set_beta(beta_t, t - 1)
        # Place updated_z at the t-1's position in beta list
        self.state.set_hidden_state(updated_z, time=(t - 1))
        # next_prediction shape: y_dim x 1
        self.state.set_prediction(t_prediction, t)

        return t_prediction
