import torch

from fedmoe.metrics.metric_managers import MetricManager
from fedmoe.metrics.metrics import RMSEMetric


torch.set_default_dtype(torch.float64)


def test_rmse_metric() -> None:
    input = torch.Tensor([0, 1, 2, 3, 4, 5])
    server_outputs = [
        torch.sin(input[0]),
        torch.sin(input[1]),
        torch.sin(input[2]),
        torch.sin(input[3]),
        torch.sin(input[4]),
        torch.sin(input[5]),
    ]
    true_values = [
        torch.Tensor([0.0]),
        torch.Tensor([0.84147098]),
        torch.Tensor([0.90929743]),
        torch.Tensor([0.14112001]),
        torch.Tensor([-0.7568025]),
        torch.Tensor([-0.95892427]),
    ]

    metric_manager = MetricManager(metrics=[RMSEMetric("RMSE")], metric_manager_name="server")
    for i in range(0, len(server_outputs)):
        metric_manager.update(
            {"server_predictions": torch.Tensor([server_outputs[i]])},
            torch.Tensor(true_values[i]),
        )

    # Compute metric
    final_metric_value = metric_manager.compute()
    metric_value = torch.Tensor([final_metric_value["server - server_predictions - RMSE"]])

    manual_metric_value = torch.Tensor([0.0])

    assert torch.allclose(metric_value, manual_metric_value, rtol=0.0, atol=1e-4)

    metric_manager.clear()
    # Assuming the prediction is y = 0.0
    wrong_prediction = [0.0]
    for i in range(0, len(true_values)):
        metric_manager.update(
            {"server_predictions": torch.Tensor([wrong_prediction])},
            torch.Tensor(true_values[i]),
        )

    # Compute metric
    final_metric_value = metric_manager.compute()
    metric_value = torch.Tensor([final_metric_value["server - server_predictions - RMSE"]])

    manual_absolute_values = torch.Tensor(
        [
            torch.Tensor([0.0]),
            torch.Tensor([0.84147098]),
            torch.Tensor([0.90929743]),
            torch.Tensor([0.14112001]),
            torch.Tensor([0.7568025]),
            torch.Tensor([0.95892427]),
        ]
    )
    denominator = torch.Tensor([6.0])
    manual_pow_2 = torch.sum(torch.pow(manual_absolute_values, 2) / denominator)
    manual_error = torch.sqrt(manual_pow_2)

    assert torch.allclose(metric_value, manual_error, rtol=0.0, atol=1e-4)
