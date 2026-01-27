from collections.abc import Callable

import torch


InputGenerationFunction = Callable[[torch.Tensor], torch.Tensor]
TargetGenerationFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
