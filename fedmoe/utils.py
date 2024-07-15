from enum import Enum
from typing import Tuple

import torch


class TensorGenerationType(Enum):
    UNIFORM_NEG_ONE_TO_ONE = "UNIFORM_NEG_ONE_TO_ONE"
    STANDARD_GAUSSIAN = "STANDARD_GAUSSIAN"


def generate_random_tensor(
    generator_type: TensorGenerationType, shape: Tuple[int, ...], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if generator_type is TensorGenerationType.STANDARD_GAUSSIAN:
        r_tensor = torch.randn(shape, device=device)
    elif generator_type is TensorGenerationType.UNIFORM_NEG_ONE_TO_ONE:
        r_tensor = torch.rand(shape, device=device) * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown generator type: {generator_type.value}")

    return r_tensor.double()
