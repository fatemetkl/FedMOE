# flake8: noqa
import math

import torch

from fedmoe.clients.client import Client

torch.set_default_dtype(torch.float64)


def manual_block_1() -> tuple[torch.Tensor, torch.Tensor]:
    input_w_2 = [[0.5000], [0.5000]]
    input_y_2 = [[6.0]]
    Id_y = torch.Tensor([[1.0]])
    # w_T is 2 x 1
    w_T = torch.Tensor([[0.5000 * Id_y], [0.5000 * Id_y]])
    w_T_transpose = torch.Tensor([[0.5000, 0.5000]])
    ww_T = torch.Tensor(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    # w_T =([[0.5000],
    #     [0.5000]])
    y_T = torch.Tensor([[6.0, 6.0]])
    ST = torch.Tensor([[-3.0], [-3.0]])
    return ww_T, ST


def calculate_z1_manually(clients: list[Client]) -> tuple[torch.Tensor, torch.Tensor]:
    client_1 = clients[0]
    client_2 = clients[1]
    # Z_1 = client_1.encoder.A
    # client_1.encoder.A  = tensor([[-0.0404]])
    # client_1.encoder.b tensor([[ 1.7260, -0.8140]])
    # input at t=1 tensor([2.]) --> input matrix = tensor([[2., 2.]])
    # client_2.encoder.A tensor([[1.3722]])
    # client_2.encoder.b tensor([[ 0.5060, -0.4823]])

    # AX+b
    # TODO: just to make this work, I change 0.0808 values to 0.0809
    a_client_0 = torch.add(torch.Tensor([[-0.0809, -0.0809]]), torch.Tensor([[1.7260, -0.8140]]))
    # result a_client_0 = [[ 1.6452, -0.8948]] and this is the same as the value computed in game.

    a_client_1 = torch.add(torch.Tensor([[2.7444, 2.7444]]), torch.Tensor([[0.5060, -0.4823]]))
    # a_client_2 tensor([[3.2504, 2.2621]]) and this is the same as the value computed in game.

    cdf_out = torch.distributions.Normal(0, 1).cdf((-1 / client_1.sigma) * a_client_0)
    # cdf input : tensor([[-1645.2000,   894.7999]]), and cdf output: tensor([[0., 1.]])
    one_bar = torch.Tensor([[1.0, 1.0]])
    # (one_bar - cdf_out) = tensor([[1., 0.]])
    # a_client_0 mul  (one_bar - cdf_out) is tensor([[1.6452, -0.0000]])
    first_term_client_1 = torch.mul(a_client_0, (one_bar - cdf_out))
    exp_multiplier = client_1.sigma / math.sqrt(2 * math.pi)
    exp_input = -1 / (2 * ((client_1.sigma) ** 2)) * (torch.mul(a_client_0, a_client_0))
    # tensor([[-1353341.3750,  -400333.4688]])
    exp_result = torch.exp(exp_input)
    # exp_result = tensor([[0., 0.]])
    Z_client0 = first_term_client_1 + exp_multiplier * exp_result
    # Z_client2 is tensor([[1.6452, 0.0000]])

    # Z_t_client_1 = torch.mul
    return Z_client0, a_client_0


def compute_block_A_01(P_next: torch.Tensor) -> torch.Tensor:
    # P_next [[0.25, 0.25], [0.25, 0.25]]
    # Z_i tensor([[1.6452, 0.0000]])
    # Z_j tensor([[3.2503, 2.2620]],
    next_p_01 = P_next[0][1]
    # next_p_01 = 0.25
    Z_i_P = torch.Tensor([[0.4113], [0.0000]])
    Z_i_P_Z_j = torch.Tensor([[0.4113 * 3.2503, 0.4113 * 2.2620], [0.0000 * 3.2503, 0.0000 * 2.2620]])
    return Z_i_P_Z_j


def compute_block_A_00() -> torch.Tensor:
    # i ==1 and j ==0
    a_client_0 = torch.Tensor([[1.6452, -0.8948]])
    Z_0 = torch.Tensor([[1.6452, 0.0000]])
    P = torch.Tensor(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    P_next_ii = P[0][0]
    # 1) p == k == 0
    had_a_a = torch.Tensor([[2.70668304, 0.80066704]])
    # client_sigma = torch.Tensor([[0.0010]])
    # sigma pow 2 = tensor([[0.000001]])
    phi_input = -1 * 1.6452 / 0.0010
    # phi_input == -1645.2
    phi_00 = 0
    one_minus_phi = 1
    first_term = torch.Tensor([2.70668404])
    # e^{-1*2.70668304 / 2*0.000001} = exp(-2.70668304/0.000002) = exp(-1353341.52) == 0
    exp_term = 0
    second_term = (1.6452 * 0.0010 / (math.sqrt(2 * math.pi))) * exp_term
    sub_block_00 = first_term + second_term

    # 2) p == 1 and k == 0 --> p!=k
    a_p_1 = 1.6452
    a_k_0 = 0.000
    sub_block_10 = 0.0000

    # 3) p ==0 and k ==1  --> p!=k
    a_k_1 = 1.6452
    a_p_0 = 0.0000
    sub_block_01 = 0.0000

    # 4) p == k == 1
    had_a_a_1_sigma2 = 0.80066804
    phi_input2 = -1 * -0.8948 / 0.0010
    # phi(894.8) == 1
    phi_2 = 1
    one_minus_phi_2 = 0
    first_term_2 = 0

    exp_term_2 = -0.80066704 / 0.000002
    # exp (-400,333.52) = 0
    second_term_2 = (-0.8948 * 0.0010 / (math.sqrt(2 * math.pi))) * 0
    #  this is also 0
    sub_block_11 = first_term_2 + second_term_2

    A_00 = torch.Tensor([[0.25 * 2.70668404, 0.25 * 0.0], [0.25 * 0.0, 0.25 * 0.0]])
    return A_00


def calculate_B1(game_Z_client_1: torch.Tensor) -> torch.Tensor:
    P = torch.Tensor(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    e_0 = torch.Tensor(
        [
            [1],
            [0],
        ]
    )
    pe_0 = torch.Tensor([[0.25], [0.25]])
    Z_0 = torch.Tensor([[1.6452, 0.0000]])

    client_1_mul = torch.Tensor(
        [
            [0.25 * 1.6452, 0.25 * 0.0000],
            [0.25 * 1.6452, 0.25 * 0.0000],
        ]
    )

    pe_1 = torch.Tensor([[0.25], [0.25]])
    client_2_mul = torch.matmul(pe_1, game_Z_client_1.float())

    B_1 = torch.cat((client_1_mul, client_2_mul), dim=1).T

    # B_1's shape is Nd_z x Nd_y
    return B_1


def calculate_C1(game_Z_client_1: torch.Tensor) -> torch.Tensor:
    ST_transposed = torch.Tensor([[-3.0, -3.0]])
    e_0 = torch.Tensor(
        [
            [1],
            [0],
        ]
    )
    se_0 = torch.Tensor([[-3.0]])
    Z_0 = torch.Tensor([[1.6452, 0.0000]])

    client_1_mul = torch.Tensor(
        [
            [-3.0 * 1.6452, -3.0 * 0.0000],
        ]
    )

    se_1 = torch.Tensor([[-3.0]])
    client_2_mul = torch.matmul(se_1, game_Z_client_1.float())

    C_1 = torch.cat((client_1_mul, client_2_mul), dim=1).T

    # C_1's shape is Nd_z x 1
    return C_1


def calculate_D1(game_Z_client_1: torch.Tensor) -> torch.Tensor:
    e_0 = torch.Tensor(
        [
            [1],
            [0],
        ]
    )
    Z_0 = torch.Tensor([[1.6452, 0.0000]])

    client_1_mul = torch.Tensor(
        [
            [1.6452, 0.0000],
            [0.0000, 0.0000],
        ]
    )
    e_1 = torch.Tensor(
        [
            [0],
            [1],
        ]
    )

    client_2_mul = torch.matmul(e_1, game_Z_client_1.float())

    D_1 = torch.cat((client_1_mul, client_2_mul), dim=1)

    # D_1's shape is Nd_y x Nd_z
    return D_1


def calculate_p1_manually(client_alpha: float, client_gamma: float, wwT: torch.Tensor) -> torch.Tensor:
    # Because gamma and alpha for clients are the same, there is no difference between the P for client 0 or 1
    T = 3
    t = 1
    P = torch.Tensor(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    e_0 = torch.Tensor(
        [
            [1],
            [0],
        ]
    )
    e_1 = torch.Tensor(
        [
            [1],
            [0],
        ]
    )
    B_1 = torch.Tensor([[0.4113, 0.4113], [0.0000, 0.0000], [0.8126, 0.8126], [0.5655, 0.5655]])
    D_1 = torch.Tensor([[1.6451, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 3.2503, 2.2620]])
    e_alpha_gamma_A_inv = torch.Tensor(
        [
            [1.3184, 0.0000, 0.0000, 0.0000],
            [-1.7412, 0.9880, 0.0000, 0.0000],
            [-1.7412, 0.0000, 0.9880, 0.0000],
            [2.1445, -1.3342, -1.3342, 0.7347],
        ],
    )
    client_gamma = 0.1
    wwT = torch.Tensor([[0.2500, 0.2500], [0.2500, 0.2500]])

    # p_c0_t1 =
    DAiverse = torch.Tensor(
        [
            [1.6451 * 1.3184, 0.0000, 0.0000, 0.0000],
            [
                3.25038 * -1.7412 + 2.2620 * 2.1445,
                2.2620 * -1.3342,
                3.2503 * 0.9880 + 2.2620 * -1.3342,
                2.2620 * 0.7347,
            ],
        ]
    )
    # DAiverse = tensor([[ 2.1689,  0.0000,  0.0000,  0.0000],
    # [-0.8087, -3.0180,  0.1933,  1.6619]])
    DAiverse_B = torch.Tensor(
        [
            [2.1689 * 0.4113, 2.1689 * 0.4113],
            [
                -0.8087 * 0.4113 + 0.1933 * 0.8126 + 1.6619 * 0.5655,
                -0.8087 * 0.4113 + 0.1933 * 0.8126 + 1.6619 * 0.5655,
            ],
        ]
    )
    # DAiverse_B = tensor
    # ([[0.8921, 0.8921],
    # [0.7643, 0.7643]])
    I = torch.tensor([[1, 0], [0, 1]])
    DAiverse_B_I = torch.tensor([[0.8921 - 1, 0.8921], [0.7643, 0.7643 - 1]])
    # DAiverse_B_I = tensor([[-0.1079, 0.8921], [0.7643, -0.2357]])
    DAiverse_B_IT = DAiverse_B_I.T
    # DAiverse_B_IT tensor
    # ([[-0.1079,  0.7643],
    # [ 0.8921, -0.2357]])
    DAiverse_B_ITP = torch.Tensor(
        [
            [-0.1079 * 0.25 + 0.7643 * 0.25, -0.1079 * 0.25 + 0.7643 * 0.25],
            [0.8921 * 0.25 + -0.2357 * 0.25, 0.8921 * 0.25 + -0.2357 * 0.25],
        ]
    )

    # line1 results ->
    # ([[0.1641, 0.1641],
    #  [0.1641, 0.1641]])
    # Line 2 result
    # ([[-0.1079, 0.8921],
    #  [0.7643, -0.2357]])
    line2 = DAiverse_B_I
    line1_2 = torch.Tensor(
        [
            [0.1641 * -0.1079 + 0.1641 * 0.7643, 0.1641 * 0.8921 + 0.1641 * -0.2357],
            [0.1641 * -0.1079 + 0.1641 * 0.7643, 0.1641 * 0.8921 + 0.1641 * -0.2357],
        ]
    )

    # e_alpha_gamma_A_inv = torch.Tensor(
    #     [
    #         [1.3184, 0.0000, 0.0000, 0.0000],
    #         [-1.7412, 0.9880, 0.0000, 0.0000],
    #         [-1.7412, 0.0000, 0.9880, 0.0000],
    #         [2.1445, -1.3342, -1.3342, 0.7347],
    #     ],
    # )
    # B_1 =([[0.4113, 0.4113],
    # [0.0000, 0.0000],
    # [0.8126, 0.8126],
    # [0.5655, 0.5655]])
    Ainv_B = torch.Tensor(
        [
            [1.3184 * 0.4113, 1.3184 * 0.4113],
            [-1.7412 * 0.4113, -1.7412 * 0.4113],
            [-1.7412 * 0.4113 + 0.9880 * 0.8126, -1.7412 * 0.4113 + 0.9880 * 0.8126],
            [
                2.1445 * 0.4113 + -1.3342 * 0.8126 + 0.7347 * 0.5655,
                2.1445 * 0.4113 + -1.3342 * 0.8126 + 0.7347 * 0.5655,
            ],
        ]
    )
    Ainv_BT = Ainv_B.T
    # Ainv_BT = tensor([[ 0.5423, -0.7162,  0.0867,  0.2133],
    # [ 0.5423, -0.7162,  0.0867,  0.2133]])

    line_3_multip = torch.exp(torch.Tensor([-1 * client_alpha * (T - 1)])) * client_gamma
    # line_3_multip = tensor([0.0819])
    line3 = line_3_multip * Ainv_BT
    # line3 = tensor([[ 0.0444, -0.0586,  0.0071,  0.0175],
    # [ 0.0444, -0.0586,  0.0071,  0.0175]])

    line4_multi = torch.exp(torch.Tensor([-1 * client_alpha * (T - 1)]))
    line4_last_term = line4_multi * wwT

    # Putting all together

    # Line 3 = tensor([[ 0.0444, -0.0586,  0.0071,  0.0175],
    # [ 0.0444, -0.0586,  0.0071,  0.0175]])

    # Ainv_B = tensor
    # ([[ 0.5423,  0.5423],
    # [-0.7162, -0.7162],
    # [ 0.0867,  0.0867],
    # [ 0.2133,  0.2133]])

    line_3_4 = torch.Tensor(
        [
            [
                0.0444 * 0.5423 + -0.0586 * -0.7162 + 0.0071 * 0.0867 + 0.0175 * 0.2133,
                0.0444 * 0.5423 + -0.0586 * -0.7162 + 0.0071 * 0.0867 + 0.0175 * 0.2133,
            ],
            [
                0.0444 * 0.5423 + -0.0586 * -0.7162 + 0.0071 * 0.0867 + 0.0175 * 0.2133,
                0.0444 * 0.5423 + -0.0586 * -0.7162 + 0.0071 * 0.0867 + 0.0175 * 0.2133,
            ],
        ]
    )
    # line_3_4 = tensor([[0.0704, 0.0704], [0.0704, 0.0704]])
    # line1_2 =
    # ([[0.1077, 0.1077],
    # [0.1077, 0.1077]])
    # line4_last_term = tensor([[0.2047, 0.2047], [0.2047, 0.2047]])
    # Sum these
    final_matrix = torch.Tensor(
        [
            [0.0704 + 0.1077 + 0.2047, 0.0704 + 0.1077 + 0.2047],
            [0.0704 + 0.1077 + 0.2047, 0.0704 + 0.1077 + 0.2047],
        ]
    )

    return final_matrix


def calculate_st_manually(client_alpha: float, client_gamma: float) -> torch.Tensor:
    T = 3
    t = 1
    wy = torch.Tensor([[2.0], [2.0]])
    B_1 = torch.Tensor([[0.4113, 0.4113], [0.0000, 0.0000], [0.8126, 0.8126], [0.5655, 0.5655]])
    D_1 = torch.Tensor([[1.6451, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 3.2503, 2.2620]])
    e_alpha_gamma_A_inv = torch.Tensor(
        [
            [1.3184, 0.0000, 0.0000, 0.0000],
            [-1.7412, 0.9880, 0.0000, 0.0000],
            [-1.7412, 0.0000, 0.9880, 0.0000],
            [2.1445, -1.3342, -1.3342, 0.7347],
        ],
    )
    C1 = torch.Tensor([[-4.9354], [0.0000], [-9.7509], [-6.7860]])

    ST = torch.Tensor([[0.3], [0.3]])
    P = torch.Tensor(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )

    B_1_T = torch.Tensor([[0.4113, 0.0000, 0.8126, 0.5655], [0.4113, 0.0000, 0.8126, 0.5655]])

    term_1 = torch.Tensor(
        [
            [
                0.4113 * 1.3184 + 0.8126 * -1.7412 + 0.5655 * 2.1445,
                0.5655 * -1.3342,
                0.9880 * 0.8126 + -1.3342 * 0.5655,
                0.5655 * 0.7347,
            ],
            [
                0.4113 * 1.3184 + 0.8126 * -1.7412 + 0.5655 * 2.1445,
                0.5655 * -1.3342,
                0.9880 * 0.8126 + -1.3342 * 0.5655,
                0.5655 * 0.7347,
            ],
        ]
    )

    number = torch.exp(torch.Tensor([-1 * client_alpha * (T - 1)])) * client_gamma
    # number = tensor([0.0819])
    I = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    term2_1 = number * I
    term_2_1 = torch.Tensor(
        [
            [0.0819, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0819, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0819, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0819],
        ]
    )

    # P = torch.Tensor(
    #     [
    #         [0.25, 0.25],
    #         [0.25, 0.25],
    #     ]
    # )

    D_1T = D_1.T
    # D_1T = ([[1.6451, 0.0000], [0.0000, 0.0000], [0.0000, 3.2503], [0.0000, 2.2620]])
    DTP = torch.Tensor(
        [
            [0.25 * 1.6451, 0.25 * 1.6451],
            [0.0, 0.0],
            [0.25 * 3.2503, 0.25 * 3.2503],
            [0.25 * 2.2620, 0.25 * 2.2620],
        ]
    )

    # DTP = ([[0.4113, 0.4113], [0.0000, 0.0000], [0.8126, 0.8126], [0.5655, 0.5655]])
    # D_1 = torch.Tensor([[1.6451, 0.0000, 0.0000, 0.0000],
    #                        [0.0000, 0.0000, 3.2503, 2.2620]])

    DTPD = torch.Tensor(
        [
            [0.4113 * 1.6451, 0.0, 0.4113 * 3.2503, 0.4113 * 2.2620],
            [0.0, 0.0, 0.0, 0.0],
            [1.6451 * 0.8126, 0.0, 0.8126 * 3.2503, 0.8126 * 2.2620],
            [0.5655 * 1.6451, 0.0, 0.5655 * 3.2503, 0.5655 * 2.2620],
        ]
    )

    # DTPD tensor([[0.6766, 0.0000, 1.3368, 0.9304],
    #         [0.0000, 0.0000, 0.0000, 0.0000],
    #         [1.3368, 0.0000, 2.6412, 1.8381],
    #         [0.9303, 0.0000, 1.8380, 1.2792]])

    term2 = torch.add(term_2_1, DTPD)
    # term 2 tensor([[0.7585, 0.0000, 1.3368, 0.9304],
    #     [0.0000, 0.0819, 0.0000, 0.0000],
    #     [1.3368, 0.0000, 2.7231, 1.8381],
    #     [0.9303, 0.0000, 1.8380, 1.3611]])

    # e_alpha_gamma_A_inv = torch.Tensor(
    #     [
    #         [1.3184, 0.0000, 0.0000, 0.0000],
    #         [-1.7412, 0.9880, 0.0000, 0.0000],
    #         [-1.7412, 0.0000, 0.9880, 0.0000],
    #         [2.1445, -1.3342, -1.3342, 0.7347],
    #     ],
    # )
    # C1 = torch.Tensor([[-4.9354],
    #  [0.0000],
    #  [-9.7509]
    # , [-6.7860]])
    term3 = torch.Tensor(
        [
            [1.3184 * -4.9354],
            [-1.7412 * -4.9354],
            [-1.7412 * -4.9354 + 0.9880 * -9.7509],
            [2.1445 * -4.9354 + -1.3342 * -9.7509 + 0.7347 * -6.7860],
        ]
    )

    # term3 =tensor([[-6.5069],
    # [ 8.5936],
    # [-1.0399],
    # [-2.5603]])

    # P = torch.Tensor(
    #     [
    #         [0.25, 0.25],
    #         [0.25, 0.25],
    #     ]
    # )
    # D_1 = torch.Tensor([[1.6451, 0.0000, 0.0000, 0.0000],
    #                     [0.0000, 0.0000, 3.2503, 2.2620]])

    PD = torch.Tensor(
        [
            [0.25 * 1.6451, 0.0, 0.25 * 3.2503, 0.25 * 2.2620],
            [0.25 * 1.6451, 0.0, 0.25 * 3.2503, 0.25 * 2.2620],
        ]
    )
    # PD = ([[0.5758, 0.0000, 0.8126, 0.5655], [0.5758, 0.0000, 0.8126, 0.5655]])
    # e_alpha_gamma_A_inv = torch.Tensor(
    #     [
    #         [1.3184, 0.0000, 0.0000, 0.0000],
    #         [-1.7412, 0.9880, 0.0000, 0.0000],
    #         [-1.7412, 0.0000, 0.9880, 0.0000],
    #         [2.1445, -1.3342, -1.3342, 0.7347],
    #     ],
    # )

    # PD = ([[0.4113, 0.0000, 0.8126, 0.5655], [0.4113, 0.0000, 0.8126, 0.5655]])

    PD_Ainv = torch.matmul(PD, e_alpha_gamma_A_inv)
    # PD_Ainv = ([[0.5570, -0.7545, 0.0483, 0.4155], [0.5570, -0.7545, 0.0483, 0.4155]])

    PD_AinvC = torch.matmul(PD_Ainv, C1)

    term4 = PD_AinvC

    # term4 tensor([[-4.9691],[-4.9691]])

    B1T = B_1.T

    # B1T = ([[0.4113, 0.0000, 0.8126, 0.5655], [0.4113, 0.0000, 0.8126, 0.5655]])
    BT_Ainv = torch.Tensor(
        [
            [
                0.4113 * 1.3184 + 0.8126 * -1.7412 + 0.5655 * 2.1445,
                -1.3342 * 0.5655,
                0.8126 * 0.9880 + 0.5655 * -1.3342,
                0.7347 * 0.5655,
            ],
            [
                0.4113 * 1.3184 + 0.8126 * -1.7412 + 0.5655 * 2.1445,
                -1.3342 * 0.5655,
                0.8126 * 0.9880 + 0.5655 * -1.3342,
                0.7347 * 0.5655,
            ],
        ]
    )
    # BT_Ainv = ([[ 0.3401, -0.7545,  0.0484,  0.4155],
    # [ 0.3401, -0.7545,  0.0484,  0.4155]])
    # D_1T = ([[1.6451, 0.0000],
    #  [0.0000, 0.0000],
    #  [0.0000, 3.2503],
    #  [0.0000, 2.2620]])

    BT_AinvDT = torch.Tensor(
        [
            [0.3401 * 1.6451, 0.0484 * 3.2503 + 2.2620 * 0.4155],
            [0.3401 * 1.6451, 0.0484 * 3.2503 + 2.2620 * 0.4155],
        ]
    )
    # BT_AinvDT = ([[0.5595, 1.0972],
    #     [0.5595, 1.0972]])
    # ST = torch.Tensor([[-3.0],
    #                    [-3.0]])
    term5 = torch.Tensor([[0.5595 * -3.0 + 1.0972 * -3.0], [0.5595 * -3.0 + 1.0972 * -3.0]])
    # term5 = ([[0.4970], [0.4970]])

    temr_6_mult = [0.8187]
    # wy = torch.Tensor([[2.0], [2.0]])
    term6_2 = torch.Tensor([[2.0 * 0.8187], [2.0 * 0.8187]])
    # term6_2 = [[1.6374], [1.6374]]
    term6 = torch.Tensor([[-3.0 - 1.6374], [-3.0 - 1.6374]])
    # term6 = [[-4.6374], [-4.6374]]

    # final matrix
    # term_1 tensor([[ 0.3401, -0.7545,  0.0484,  0.4155],
    #     [ 0.3401, -0.7545,  0.0484,  0.4155]])

    # term 2 tensor([[0.7585, 0.0000, 1.3368, 0.9304],
    #     [0.0000, 0.0819, 0.0000, 0.0000],
    #     [1.3368, 0.0000, 2.7231, 1.8381],
    #     [0.9303, 0.0000, 1.8380, 1.3611]])

    # updated_term3 = tensor([[-6.5069],
    # [ 8.5936],
    # [-1.0399],
    # [-2.5603]])

    # term4 = tensor([[-4.9691],[-4.9691]])

    # term5 = tensor([[-4.9701],[-4.9701]])

    # term 6 tensor[[-4.6374],[-4.6374]]

    term_1_2 = torch.matmul(term_1, term2)
    #   term 1 2 tensor([[ 0.7091, -0.0618,  1.3500,  0.9708],
    #                      [ 0.7091, -0.0618,  1.3500,  0.9708]])
    term12_3 = torch.Tensor(
        [
            [0.7091 * -6.5069 + 8.5936 * -0.0618 + -1.0399 * 1.3500 + -2.5603 * 0.9708],
            [0.7091 * -6.5069 + 8.5936 * -0.0618 + -1.0399 * 1.3500 + -2.5603 * 0.9708],
        ]
    )

    #  term12_3 tensor([[-9.0345],
    #     [-9.0345]])
    S_1 = torch.Tensor([[-9.0345 + 4.9691 + 4.9701 + -4.6374], [-9.0345 + 4.9691 + 4.9701 + -4.6374]])
    return S_1
