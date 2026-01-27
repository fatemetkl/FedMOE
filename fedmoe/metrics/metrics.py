from abc import ABC, abstractmethod

import sklearn
import torch


torch.set_default_dtype(torch.float64)


class SimpleMetric(ABC):
    def __init__(self, name: str) -> None:
        """
        Abstract metric class with base functionality to update, compute and clear metrics. User needs to define
        ``__call__`` method which returns metric given inputs and target.

        Args:
            name (str): Name of the metric.
        """
        self.name = name
        self.accumulated_inputs: list[torch.Tensor] = []
        self.accumulated_targets: list[torch.Tensor] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        This method updates the state of the metric by appending the passed input and target pairing to their
        respective list.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.
        """
        self.accumulated_inputs.append(input)
        self.accumulated_targets.append(target)

    def compute(self, name: str | None = None) -> dict[str, float]:
        """
        Compute metric on accumulated input and output over updates.

        Args:
            name (str | None): Optional name used in conjunction with class attribute name to define key in metrics
                dictionary.

        Raises:
            AssertionError: Input and target lists must be non empty.

        Returns:
            (Metrics): A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
        """
        assert len(self.accumulated_inputs) > 0 and len(self.accumulated_targets) > 0
        stacked_inputs = torch.cat(self.accumulated_inputs)
        stacked_targets = torch.cat(self.accumulated_targets)
        result = self.__call__(stacked_inputs, stacked_targets)
        result_key = f"{name} - {self.name}" if name is not None else self.name

        return {result_key: result}

    def clear(self) -> None:
        """Resets metrics by clearing input and target lists."""
        self.accumulated_inputs = []
        self.accumulated_targets = []

    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> float:
        """
        User defined method that calculates the desired metric given the predictions and target.

        Raises:
            NotImplementedError: User must define this method.
        """
        raise NotImplementedError


class RMSEMetric(SimpleMetric):
    def __init__(self, name: str = "RMSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        return sklearn.metrics.root_mean_squared_error(target, logits)


class MSEMetric(SimpleMetric):
    def __init__(self, name: str = "MSE"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        assert logits.shape[0] == target.shape[0]
        # Assuming data is batch-first
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        return sklearn.metrics.mean_squared_error(
            target,
            logits,
        )
