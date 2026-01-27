import copy
from collections.abc import Sequence

import torch

from fedmoe.metrics.metrics import SimpleMetric


class MetricManager:
    def __init__(self, metrics: Sequence[SimpleMetric], metric_manager_name: str) -> None:
        """
        Class to manage a set of metrics associated to a given prediction type.

        Args:
            metrics (Sequence[Metric]): List of metric to evaluate predictions on.
            metric_manager_name (str): Name of the metric manager (i.e. train, val, test)
        """
        self.original_metrics = metrics
        self.metric_manager_name = metric_manager_name
        self.metrics_per_prediction_type: dict[str, Sequence[SimpleMetric]] = {}

    def update(self, preds: dict[str, torch.Tensor], target: torch.Tensor) -> None:
        """
        Updates (or creates then updates) a list of metrics for each prediction type.

        Args:
            preds (dict[str, torch.Tensor]): A dictionary of preds from the model.
            target (torch.Tensor): The ground truth labels for the data. All elements of the dictionary will be
                computed against this target.
        """
        if not self.metrics_per_prediction_type:
            self.metrics_per_prediction_type = {key: copy.deepcopy(self.original_metrics) for key in preds}

        for prediction_key, pred in preds.items():
            metrics_for_prediction_type = self.metrics_per_prediction_type[prediction_key]
            assert len(preds) == len(self.metrics_per_prediction_type)
            for metric_for_prediction_type in metrics_for_prediction_type:
                metric_for_prediction_type.update(pred, target)

    def compute(self) -> dict[str, float]:
        """
        Computes set of metrics for each prediction type.

        Returns:
            (dict[str, float]): dictionary containing computed metrics along with string identifiers for each
                prediction type.
        """
        all_results = {}
        for metrics_key, metrics in self.metrics_per_prediction_type.items():
            for metric in metrics:
                result = metric.compute(f"{self.metric_manager_name} - {metrics_key}")
                all_results.update(result)

        return all_results

    def clear(self) -> None:
        """Clears data accumulated in each metric for each of the prediction type."""
        for metrics_for_prediction_type in self.metrics_per_prediction_type.values():
            for metric in metrics_for_prediction_type:
                metric.clear()

    def reset(self) -> None:
        """Resets the metrics to their initial state."""
        # On next update, metrics will be recopied from self.original_metrics which are still in their initial state
        self.metrics_per_prediction_type = {}
