import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn

from fedmoe.state.client_state import ClientState


torch.set_default_dtype(torch.float64)


class ClientType(Enum):
    RFN = 0
    TRANSFORMER = 1
    ESN = 2


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

    @abstractmethod
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
        init_hidden_state_neg1: torch.Tensor | None = None,
        init_prediction_0: torch.Tensor | None = None,
        init_prediction_neg1: torch.Tensor | None = None,
    ) -> None:
        if init_hidden_state_neg1 is None:
            # Initializing Z to zero rather than a random value
            init_hidden_state_neg1 = torch.zeros((self.y_dim, self.z_dim))
        if init_prediction_0 is None:
            # Initializing with zero rather than a random value
            # init_prediction_0 = torch.zeros((self.y_dim, 1))
            init_prediction_0 = self.get_y(0)
        if init_prediction_neg1 is None:
            # Initializing with zero rather than a random value
            init_prediction_neg1 = torch.zeros((self.y_dim, 1))
        assert (
            init_prediction_0 is not None and init_prediction_neg1 is not None and init_hidden_state_neg1 is not None
        )

        self.state.init_state(
            self.z_dim,
            self.y_dim,
            init_hidden_state_neg1,
            init_prediction_0,
            init_prediction_neg1,
        )

    def init_p_s(self, num_clients: int, game_T: int) -> None:
        self.P = torch.zeros(
            game_T + 1,  # Game records P values for 0 to T inclusive.
            num_clients * self.y_dim,
            num_clients * self.y_dim,
        )
        self.S = torch.zeros(game_T + 1, num_clients * self.y_dim, 1)
        self.D = torch.zeros(
            game_T,  # D_i is only calculated from T-1 to 0
            num_clients * self.z_dim,
            num_clients * self.z_dim,
        )

    def get_x(self, t: int) -> torch.Tensor:
        return self._current_sequence[t].reshape(-1, 1)

    def get_input_matrix(self, t: int) -> torch.Tensor:
        x = self.get_x(t)
        # The input should be a 2D tensor of dimension x_dim x 1.
        assert x.shape == (self.x_dim, 1)
        # Repeating the input by columns to expand into the latent space dimension. Should result in a x_dim x z_dim
        # tensor for encoding.
        input_matrix = x.repeat(1, self.z_dim)
        assert input_matrix.shape == (self.x_dim, self.z_dim)
        return input_matrix

    def get_y(self, t: int) -> torch.Tensor:
        return self._target[t].reshape(self.y_dim, 1)

    def get_e(self, num_clients: int) -> torch.Tensor:
        e = torch.nn.functional.one_hot(torch.tensor(self.id), num_clients)
        # Creates a BLOCK column vector where each block is a y_dim x y_dim matrix. The self.id^th row is the identity
        # matrix of dim y_dim x y_dim
        bold_e_i = torch.kron(e, torch.eye(self.y_dim)).T
        assert bold_e_i.shape == (num_clients * self.y_dim, self.y_dim)
        return bold_e_i

    def get_hat_e(self, num_clients: int) -> torch.Tensor:
        e = torch.nn.functional.one_hot(torch.tensor(self.id), num_clients)
        # Creates a BLOCK column vector where each block is a z_dim x z_dim matrix. The self.id^th row is the identity
        # matrix of dim z_dim x z_dim
        bold_hat_e_i = torch.kron(e, torch.eye(self.z_dim)).T
        assert bold_hat_e_i.shape == (num_clients * self.z_dim, self.z_dim)
        return bold_hat_e_i

    def compute_X_t(self, t: int) -> torch.Tensor:
        X = []
        # t-T is the distance from t to the previous sync point (previous T).
        start_point = max(t - self.sync_steps, -1)
        # From t-T (or -1) to t-1
        for s in range(start_point, t):
            X.append(
                torch.mul(
                    pow(math.e, -1 * self.alpha * ((t - 1 - s) / 2)),
                    self.state.get_hidden_state_t(s).T,
                )
            )
        # output_X
        return torch.cat(X, dim=1).T

    def compute_y_t(self, t: int) -> torch.Tensor:
        y = []
        start_point = max(t - self.sync_steps, -1)
        for s in range(start_point, t):
            # target has shape time x y_dim so need to transform to column vector after indexing (occurs in get_y)
            residual = self.get_y(s + 1) - self.state.get_prediction_t(s)
            assert residual.shape == (self.y_dim, 1)
            y.append(pow(math.e, -1 * self.alpha * ((t - 1 - s) / 2)) * residual.T)

        # output_y
        return torch.cat(y, dim=1).T

    def update_prediction_with_beta(self, t: int, nash_beta: torch.Tensor) -> torch.Tensor:
        # Replace previous beta
        self.state.replace_beta_t(nash_beta, t)
        # Use the previous Z
        # Update prediction based on Z_t and beta_t
        next_prediction = self.state.get_prediction_t((t)) + torch.matmul(self.state.get_hidden_state_t(t), nash_beta)
        # next_prediction shape: y_dim*1
        assert next_prediction.shape == (self.y_dim, 1)
        # self.state.replace_prediction_t(next_prediction, (t+1))
        # next_prediction is \hat{Y}_t+1
        return next_prediction

    def optimize_beta(self, t: int) -> torch.Tensor:
        # Update X_t
        X_t = self.compute_X_t(t)
        y_t = self.compute_y_t(t)

        X_t_T = X_t.T
        identity_matrix = torch.eye(self.z_dim)
        first_term = torch.matmul(X_t_T, X_t) + self.gamma * identity_matrix
        second_term = torch.matmul(torch.inverse(first_term), X_t_T)

        # beta_t
        return torch.matmul(second_term, y_t)

    def predict(self, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First optimize betas based on past observations
        beta_t = self.optimize_beta(t)

        # Generate Random State and update Hidden State
        # Sequence has shape time x x_dim, but encoders expect input for a given step of x_dim x 1 (occurs in get_x)
        updated_z = self.feed_encoder(self.get_x(t))

        # Update prediction based on Z_t and beta_t
        # Having \hat{Y}_t we want to predict \hat{Y}_t+1
        # next_prediction = self.state.get_prediction_t(t) + torch.matmul(updated_z, beta_t)
        next_prediction = self.state.get_prediction_t(t) + torch.matmul(updated_z, beta_t)

        return beta_t, updated_z, next_prediction

    def update_expert(self, t: int) -> torch.Tensor:
        self.state.next_time_step(next_time=t)
        assert self.state.get_current_time() == t, "Error: time step mismatch"

        # Make prediction for time t
        beta_t, updated_z, next_prediction = self.predict(t)

        # Update State
        # Place beta_t at the t-1's position in beta list
        self.state.set_beta(beta_t, t)
        # Place updated_z at the t-1's position in beta list
        self.state.set_hidden_state(updated_z, time=t)
        # next_prediction shape: y_dim x 1
        self.state.set_prediction(next_prediction, (t + 1))

        return next_prediction
