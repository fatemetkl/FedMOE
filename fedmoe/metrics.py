import sklearn
import torch
from fl4health.utils.metrics import SimpleMetric  # type: ignore
from flwr.common.typing import Scalar


torch.set_default_dtype(torch.float64)


class RMSEMetric(SimpleMetric):
    def __init__(self, name: str = "RMSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        return sklearn.metrics.root_mean_squared_error(target, logits)


class MSEMetric(SimpleMetric):
    def __init__(self, name: str = "MSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        # Assuming data is batch-first
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        return sklearn.metrics.mean_squared_error(
            target,
            logits,
        )
