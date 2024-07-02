import torch
from fl4health.utils.metrics import SimpleMetric  # type: ignore
from flwr.common.typing import Scalar
from sklearn import metrics as sklearn_metrics  # type: ignore


class RMSEMetric(SimpleMetric):
    def __init__(self, name: str = "RMSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        return sklearn_metrics.root_mean_squared_error(target.cpu().detach(), logits.cpu().detach())


class MSEMetric(SimpleMetric):
    def __init__(self, name: str = "MSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        print(logits.shape[1], target.shape[1])
        # Assuming data is batch-first
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        return sklearn_metrics.mean_squared_error(
            target,
            logits,
        )
