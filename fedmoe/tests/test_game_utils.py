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
    """
    This function computes the regret objective. if backward_time_length is 1, we only compute regret based on
    last step (sync step), if backward_time_length == 2, previous discounted regrets are also computed and added.
    For example, backward_time_length == 2 goes one step before the sync step.
    betas are always one step before the target seq and server prediction. (beta_{t-1}, Y_{t}, y_{t})
    To calculate the regret for sync step we use (beta_{T-1}, Y_{T}, y_{T})
    """
    # Equation 9

    assert len(past_betas_i) == len(target_seq) == len(server_pred_seq) == backward_time_length
    sync_step_residual = target_seq[0] - server_pred_seq[0]
    residual_inner_product = torch.pow(torch.linalg.norm(sync_step_residual), 2.0)
    T_regularizer = gamma * torch.pow(torch.linalg.norm(past_betas_i[0]), 2.0)
    past_losses = 0
    if backward_time_length > 1:
        for past_time_i in range(1, backward_time_length):
            # e^{-alpha * past_time_i} --> the further past we go, the smaller our discounting factor gets
            residual = target_seq[past_time_i] - server_pred_seq[past_time_i]
            past_residual_inner_product = torch.pow(torch.linalg.norm(residual), 2.0)
            regularizer = gamma * torch.pow(torch.linalg.norm(past_betas_i[past_time_i]), 2.0)
            # past_time_i = T-t
            discount = pow(math.e, -1 * alpha * past_time_i)
            past_losses += discount * (past_residual_inner_product + regularizer)

    return past_losses + (residual_inner_product + T_regularizer)


def test_regret_objective_function() -> None:
    # T = 3
    # Data function: y = x1+x2
    #  We have two clients
    # example_input = torch.Tensor([[1, 2], [2, 3], [1, 4], [1, 0]])
    example_output = torch.Tensor([[3], [5], [5], [1]])
    client_gamma = 0.1
    # client_alpha = 1.0

    seed = 2024
    torch.manual_seed(seed)

    # t = 2

    # clients_pred_3 = torch.Tensor([[0.5], [1]])
    # w_2 = torch.Tensor([[0.3, 0.7]])
    server_output_3 = torch.Tensor([0.15 + 0.7])
    # residual_3 = torch.Tensor([[1 - 0.85]])
    random_betas_2 = torch.randn((2, 3, 1)).double()
    regret_3_0 = (0.15) ** 2 + client_gamma * torch.pow(torch.linalg.norm(random_betas_2[0]), 2.0)
    regret_3_1 = (0.15) ** 2 + client_gamma * torch.pow(torch.linalg.norm(random_betas_2[1]), 2.0)

    # t = 1
    # clients_pred_2 = torch.Tensor([[4.8], [5.1]])
    # w_2 = torch.Tensor([[0.4, 0.6]])
    server_output_2 = torch.Tensor([4.8 * 0.4 + 5.1 * 0.6])
    # residual_2 = torch.Tensor([[5 - 4.98]])

    random_betas_1 = torch.randn((2, 3, 1)).double()
    regret_2_0 = pow(math.e, -1.0) * (
        (0.02) ** 2 + client_gamma * torch.pow(torch.linalg.norm(random_betas_1[0]), 2.0)
    )

    regret_2_1 = pow(math.e, -1.0) * (
        (0.02) ** 2 + client_gamma * torch.pow(torch.linalg.norm(random_betas_1[1]), 2.0)
    )

    #  regret for the last two steps (T=2)
    #  Calculations using the function

    computed_regret_c0 = compute_game_regret_objective(
        [random_betas_2[0], random_betas_1[0]],
        [torch.Tensor(example_output[3]), torch.Tensor(example_output[2])],
        [server_output_3, server_output_2],
        gamma=0.1,
        alpha=1.0,
        backward_time_length=2,
    )

    computed_regret_c1 = compute_game_regret_objective(
        [random_betas_2[1], random_betas_1[1]],
        [torch.Tensor(example_output[3]), torch.Tensor(example_output[2])],
        [server_output_3, server_output_2],
        gamma=0.1,
        alpha=1.0,
        backward_time_length=2,
    )
    manual_regret = regret_3_0 + regret_3_1 + regret_2_0 + regret_2_1
    assert torch.allclose(manual_regret, (computed_regret_c1 + computed_regret_c0), rtol=0.0, atol=1e-5)
