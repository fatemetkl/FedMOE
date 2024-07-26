import math
from typing import List

import torch


def compute_game_regret_objective(
    past_betas_i: List[torch.Tensor],
    target_seq: List[torch.Tensor],
    server_pred_seq: List[torch.Tensor],
    gamma: float,
    alpha: float,
    backward_time_length: int,
) -> torch.Tensor:
    # Equation 9
    # First index in each input sequences corresponds to the value for T
    # If backward is 0, we just consider T, but if it is more, we go back
    sync_step_residual = target_seq[0] - server_pred_seq[0]
    residual_inner_product = torch.pow(torch.linalg.norm(sync_step_residual), 2.0)
    T_regularizer = gamma * torch.pow(torch.linalg.norm(past_betas_i[0]), 2.0)
    past_losses = 0
    if backward_time_length > 0:
        for past_time_i in (1, backward_time_length + 1):
            print(past_time_i)
            # e^{-alpha * past_time_i} --> the further past we go, the smaller our discounting factor gets
            residual = target_seq[past_time_i] - server_pred_seq[past_time_i]
            past_residual_inner_product = torch.pow(torch.linalg.norm(residual), 2.0)
            regularizer = gamma * torch.pow(torch.linalg.norm(past_betas_i[past_time_i]), 2.0)
            # past_time_i = T-t
            discount = pow(math.e, -1 * alpha * past_time_i)
            past_losses += discount * (past_residual_inner_product + regularizer)

    return past_losses + (residual_inner_product + T_regularizer)
